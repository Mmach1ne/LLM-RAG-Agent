import os
import json
import sqlite3
import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
import random
from collections import defaultdict
import uuid
import time
import math
import difflib
import asyncio
from pathlib import Path
import traceback

# Google Gemini
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Vector Database & Embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Web Framework
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Document Processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# Caching
import redis
from functools import lru_cache

# Async file handling
import aiofiles

# Configuration for small VM deployment
class Config:
    # Gemini Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "LOL")
    GEMINI_MODEL = "gemini-1.5-flash"  # Lighter model for small VMs
    
    # Vector DB Configuration
    CHROMA_PERSIST_DIR = "./chroma_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight embedding model
    
    # Redis Configuration (optional, for caching)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"
    
    # Performance Settings for Small VMs
    MAX_CHUNK_SIZE = 500  # Smaller chunks for memory efficiency
    MAX_CHUNKS_PER_QUERY = 5  # Limit context size
    BATCH_SIZE = 16  # Smaller batch for embeddings
    MAX_MEMORY_ITEMS = 1000  # Limit memory storage
    
    # API Settings
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB max file size

# Previous Enums and Dataclasses (keeping them)
class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    EXECUTING = "executing"
    RETRIEVING = "retrieving"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    id: str
    description: str
    priority: Priority
    created_at: datetime.datetime
    completed: bool = False
    result: Optional[Any] = None
    dependencies: List[str] = field(default_factory=list)

@dataclass
class Memory:
    id: str
    content: Dict[str, Any]
    timestamp: datetime.datetime
    category: str
    relevance_score: float = 1.0
    embedding: Optional[List[float]] = None

# Vector Database Manager
class VectorStore:
    def __init__(self, persist_directory: str = Config.CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = Config.BATCH_SIZE):
        """Add documents to vector store in batches"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            texts = [doc["content"] for doc in batch]
            metadatas = [doc.get("metadata", {}) for doc in batch]
            ids = [doc.get("id", str(uuid.uuid4())) for doc in batch]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, batch_size=batch_size).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
    
    def search(self, query: str, k: int = Config.MAX_CHUNKS_PER_QUERY) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count())
        )
        
        if not results['documents'][0]:
            return []
        
        return [
            {
                "content": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]

# Gemini Integration
class GeminiRAG:
    def __init__(self, api_key: str = Config.GEMINI_API_KEY):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            Config.GEMINI_MODEL,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
    def generate_response(self, prompt: str, context: List[str] = None) -> str:
        """Generate response using Gemini with optional context"""
        if context:
            context_str = "\n\n".join(context)
            full_prompt = f"""Context information:
{context_str}

Based on the above context, please answer the following question:
{prompt}

If the context doesn't contain relevant information, please say so and provide a general answer."""
        else:
            full_prompt = prompt
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Document Processor
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.MAX_CHUNK_SIZE,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    async def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process uploaded file and return chunks"""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            return [
                {
                    "id": str(uuid.uuid4()),
                    "content": chunk.page_content,
                    "metadata": {
                        "source": file_path,
                        "chunk_index": i,
                        **chunk.metadata
                    }
                }
                for i, chunk in enumerate(chunks)
            ]
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

# Enhanced AI Agent with RAG
class RAGAIAgent:
    def __init__(self, name: str, personality: Dict[str, Any] = None):
        self.name = name
        self.state = AgentState.IDLE
        self.personality = personality or {
            "traits": ["helpful", "analytical", "adaptive"],
            "learning_rate": 0.1,
            "curiosity": 0.8
        }
        
        # Core components
        self.memory_bank = MemoryBank()
        self.task_manager = TaskManager()
        self.skill_registry = SkillRegistry()
        self.conversation_history = []
        self.goals = []
        self._response_times = []
        
        # RAG components
        self.vector_store = VectorStore()
        self.gemini = GeminiRAG()
        self.doc_processor = DocumentProcessor()
        
        # Cache for responses (optional Redis)
        self.cache = None
        if Config.USE_REDIS:
            try:
                self.cache = redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)
            except:
                print("Redis connection failed, proceeding without cache")
        
        # Initialize default skills
        self._register_default_skills()
        
        # Learning parameters
        self.experience_points = 0
        self.skill_levels = defaultdict(int)
    
    def get_status(self) -> Dict[str, Any]:
        if self._response_times:
            avg_rt = sum(self._response_times) / len(self._response_times)
            avg_rt_str = f"{avg_rt:.1f} ms"
        else:
            avg_rt_str = "N/A"
        return {
            "state": self.state.value,
            "active_tasks": len(self.task_manager.get_pending_tasks()),
            "memories": self.memory_bank.count(),
            "memory": {
                "database": "SQLite",
                "storage": f"{self.memory_bank.count() * 100 / 1024:.1f}KB",
                "conversations": len(self.conversation_history),
                "recallTime": "2ms",
                "count": self.memory_bank.count()
            },
            "skills": {
                skill_name.replace(" ", "_").lower(): {
                    "calls": self.skill_levels.get(skill_name, 0),
                    "enabled": True
                }
                for skill_name in self.skill_registry.list_skills()
            },
            "tasks": {
                "active": len([t for t in self.task_manager.get_pending_tasks() if not t.completed]),
                "queue": [
                    {
                        "name": task.description[:30],
                        "status": (
                            "processing"
                            if self.task_manager.task_queue
                               and task.id == self.task_manager.task_queue[0]
                            else "active"
                        )
                    }
                    for task in self.task_manager.get_pending_tasks()[:5]
                ]
            },
            "performance": {
                "avgResponseTime": avg_rt_str
            }
        }
    def record_response_time(self, start: float, end: float):
        """Call this at the end of each request/response cycle."""
        elapsed_ms = (end - start) * 1000
        self._response_times.append(elapsed_ms)
    
    def _register_default_skills(self):
        """Register built-in skills including RAG"""
        self.skill_registry.register("rag_query", self._rag_query)
        self.skill_registry.register("analyze_text", self._analyze_text)
        self.skill_registry.register("summarize", self._summarize_with_gemini)
        self.skill_registry.register("extract_entities", self._extract_entities)
        self.skill_registry.register("generate_response", self._generate_response)
        self.skill_registry.register("learn_pattern", self._learn_pattern)
        self.skill_registry.register("do_math", self._do_math)
        self.skill_registry.register("generate_code", self._generate_code_with_gemini)
        self.skill_registry.register("recall_fact", self._recall_fact)
    
    async def process_input(self, user_input: str) -> Any:
        start = time.time()
        """Process user input with RAG capabilities"""
        self.state = AgentState.PROCESSING
        
        # Check cache first
        cache_key = f"response:{hash(user_input)}"
        if self.cache:
            try:
                cached = self.cache.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"Cache error: {e}")
        
        # Create task
        task = self.task_manager.create_task(f"Process: {user_input}", Priority.MEDIUM)
        
        try:
            # Analyze intent
            intent = self._analyze_intent(user_input)
            
            # Check if we need RAG
            if self._needs_rag(user_input, intent):
                self.state = AgentState.RETRIEVING
                # await the async _rag_query
                response = await self._rag_query(user_input)
            else:
                self.state = AgentState.EXECUTING
                # Execute skill - check if it's async
                response = self._execute_skill(intent, user_input)
                
                # If response is a coroutine or Task, await it
                if asyncio.iscoroutine(response):
                    response = await response
                elif isinstance(response, asyncio.Task):
                    response = await response
            
            # Cache response
            if self.cache and isinstance(response, dict):
                try:
                    self.cache.setex(cache_key, 3600, json.dumps(response))
                except Exception as e:
                    print(f"Cache set error: {e}")
            
            # Complete task
            self.task_manager.complete_task(task.id, response)
            
            # Update conversation history
            self.conversation_history.append({
                "user": user_input,
                "agent": response
            })
            
            # Increment skill usage counter
            if isinstance(intent, dict) and "action" in intent:
                skill_name = intent["action"]
                if skill_name in self.skill_levels:
                    self.skill_levels[skill_name] += 1
                else:
                    self.skill_levels[skill_name] = 1
        
        except Exception as e:
            print(f"Error in process_input: {e}")
            traceback.print_exc()
            response = {
                "response": f"I encountered an error: {str(e)}",
                "type": "error"
            }
            self.task_manager.complete_task(task.id, response)
        
        finally:
            end = time.time()
            self.record_response_time(start, end)
            self.state = AgentState.IDLE
        
        return response
    
    def _needs_rag(self, text: str, intent: Dict[str, Any]) -> bool:
        """Determine if RAG is needed for the query"""
        rag_indicators = [
            "document", "file", "knowledge", "information", "data",
            "what does the document say", "according to", "find in",
            "search for", "look up"
        ]
        return any(indicator in text.lower() for indicator in rag_indicators)
    
    async def _rag_query(self, query: str) -> Dict[str, str]:
        """Perform RAG query"""
        # Search vector store
        relevant_docs = self.vector_store.search(query)
        
        if not relevant_docs:
            # Fall back to Gemini without context
            response = self.gemini.generate_response(query)
        else:
            # Extract content from results
            context = [doc["content"] for doc in relevant_docs[:Config.MAX_CHUNKS_PER_QUERY]]
            response = self.gemini.generate_response(query, context)
            
            # Store in memory for learning
            self.memory_bank.store(
                content={
                    "query": query,
                    "response": response,
                    "sources": [doc["metadata"].get("source", "unknown") for doc in relevant_docs]
                },
                category="rag_interaction"
            )
        
        return {"response": response, "type": "rag"}
    
    def _summarize_with_gemini(self, text: str) -> str:
        """Use Gemini for summarization"""
        prompt = f"Please provide a concise summary of the following text:\n\n{text}"
        return self.gemini.generate_response(prompt)
    
    def _generate_code_with_gemini(self, text: str) -> str:
        """Use Gemini for code generation"""
        prompt = f"Generate Python code for the following request:\n{text}\n\nProvide clean, commented code with example usage."
        response = self.gemini.generate_response(prompt)
        return f"Here's the code:\n```python\n{response}\n```"
    
    async def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest a document into the knowledge base"""
        try:
            chunks = await self.doc_processor.process_file(file_path)
            self.vector_store.add_documents(chunks)
            return {
                "success": True,
                "message": f"Successfully ingested {len(chunks)} chunks from {file_path}",
                "chunks": len(chunks)
            }
        except Exception as e:
            # Fallback for plain-text files if loader fails
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".txt":
                try:
                    async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                        text = await f.read()
                    docs = [{
                        "id": os.path.basename(file_path),
                        "content": text,
                        "metadata": {"source": os.path.basename(file_path)}
                    }]
                    self.vector_store.add_documents(docs)
                    return {
                        "success": True,
                        "message": f"Ingested plain-text file {os.path.basename(file_path)}",
                        "chunks": 1
                    }
                except Exception as fallback_e:
                    return {
                        "success": False,
                        "message": f"Error ingesting plain-text file: {str(fallback_e)}"
                    }
            return {
                "success": False,
                "message": f"Error ingesting document: {str(e)}"
            }
    
    # Keep all the original methods from the previous agent
    def _analyze_intent(self, text: str) -> Dict[str, Any]:
        # [Previous implementation remains the same]
        lower = text.lower().strip()
        
        # RAG detection
        if self._needs_rag(text, {}):
            return {
                "type": "rag",
                "action": "rag_query",
                "confidence": 0.95
            }
        
        # [Rest of the previous intent analysis code]
        # Weather detection
        weather_keywords = ["weather", "temperature", "forecast", "rain", "sunny", "cloudy", "windy", "snow", "humidity"]
        if any(word in text.lower() for word in weather_keywords):
            return {
                "type": "weather",
                "action": "weather",
                "confidence": 0.98
            }
        
        # Continue with previous implementation...
        return {
            "type": "general",
            "action": "generate_response",
            "confidence": 0.8
        }
    
    def _execute_skill(self, intent: Dict[str, Any], input_text: str) -> Any:
        """Execute skill based on intent"""
        skill_name = intent["action"]
        skill = self.skill_registry.get_skill(skill_name)
        
        if skill:
            # Handle async skills
            if asyncio.iscoroutinefunction(skill):
                return asyncio.create_task(skill(input_text))
            return skill(input_text)
        else:
            return {"response": "I'm not sure how to handle that request yet, but I'm always learning!"}
    
    # [Include all other methods from original agent]
    def _analyze_text(self, text: str) -> str:
        word_count = len(text.split())
        sentences = text.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Use Gemini for deeper analysis
        gemini_analysis = self.gemini.generate_response(
            f"Analyze the sentiment and key themes in this text: {text[:500]}"
        )
        
        return (
            f"Here's what I found about your text!\n"
            f"- Word count: {word_count}\n"
            f"- Sentences: {sentence_count}\n"
            f"- AI Analysis: {gemini_analysis}\n"
        )
    
    def _extract_entities(self, text: str) -> str:
        # Use Gemini for better entity extraction
        prompt = f"Extract all named entities (people, places, organizations, dates, numbers) from this text: {text}"
        return self.gemini.generate_response(prompt)
    
    def _generate_response(self, text: str) -> str:
        # Use Gemini for general responses
        return self.gemini.generate_response(text)
    
    def _learn_pattern(self, text: str) -> str:
        self.state = AgentState.LEARNING
        
        # Store in both memory bank and vector store
        memory_entry = {
            "text": text,
            "learned_at": datetime.datetime.now().isoformat()
        }
        
        self.memory_bank.store(
            content=memory_entry,
            category="learned_pattern"
        )
        
        # Add to vector store for RAG
        self.vector_store.add_documents([{
            "id": str(uuid.uuid4()),
            "content": text,
            "metadata": {"type": "learned_pattern", "timestamp": datetime.datetime.now().isoformat()}
        }])
        
        self.experience_points += 10
        self.state = AgentState.IDLE
        
        return f"Thanks for teaching me! I've stored that in my knowledge base: '{text}'"
    
    def _do_math(self, text: str) -> str:
        # First try local calculation
        try:
            # [Previous math implementation]
            expr = text.lower().replace('what is', '').replace('calculate', '').strip()
            # ... (rest of math processing)
            return f"The answer is: {eval(expr)}"
        except:
            # Fall back to Gemini for complex math
            return self.gemini.generate_response(f"Solve this math problem: {text}")
    
    def _recall_fact(self, text: str) -> str:
        # Search in vector store first
        results = self.vector_store.search(text, k=1)
        if results:
            return f"I remember: {results[0]['content']}"
        
        # Fall back to memory bank
        keywords = re.findall(r'\w+', text.lower())
        facts = self.memory_bank.search(keywords[0] if keywords else '', limit=1)
        if facts:
            return f"I recall: {facts[0].content.get('text', 'something about that')}"
        
        return "I don't have any information about that in my knowledge base yet."

class MemoryBank:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self.item_count = 0
    
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT,
                category TEXT,
                timestamp TEXT,
                relevance_score REAL
            )
        """)
        self.conn.commit()


    
    def store(self, content: Dict[str, Any], category: str) -> Memory:
        # Implement FIFO if exceeding max items
        if self.item_count >= Config.MAX_MEMORY_ITEMS:
            # Delete oldest entry
            self.conn.execute(
                "DELETE FROM memories WHERE id = (SELECT id FROM memories ORDER BY timestamp ASC LIMIT 1)"
            )
        
        memory = Memory(
            id=f"mem_{uuid.uuid4()}",
            content=content,
            timestamp=datetime.datetime.now(),
            category=category
        )
        
        self.conn.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?)",
            (memory.id, json.dumps(content), category, 
             memory.timestamp.isoformat(), memory.relevance_score)
        )
        self.conn.commit()
        self.item_count += 1
        
        return memory
    
    def search(self, query: str, limit: int = 5) -> List[Memory]:
        cursor = self.conn.execute(
            "SELECT * FROM memories WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit)
        )
        
        memories = []
        for row in cursor:
            memories.append(Memory(
                id=row[0],
                content=json.loads(row[1]),
                timestamp=datetime.datetime.fromisoformat(row[3]),
                category=row[2],
                relevance_score=row[4]
            ))
        
        return memories
    
    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
    
    def create_task(self, description: str, priority: Priority = Priority.MEDIUM, 
                   dependencies: List[str] = None) -> Task:
        task = Task(
            id=f"task_{datetime.datetime.now().timestamp()}",
            description=description,
            priority=priority,
            created_at=datetime.datetime.now(),
            dependencies=dependencies or []
        )
        
        self.tasks[task.id] = task
        self._update_queue()
        
        return task
    
    def complete_task(self, task_id: str, result: Any = None):
        if task_id in self.tasks:
            self.tasks[task_id].completed = True
            self.tasks[task_id].result = result
            self._update_queue()
    
    def get_pending_tasks(self) -> List[Task]:
        pending = [t for t in self.tasks.values() if not t.completed]
        return sorted(pending, key=lambda t: t.priority.value, reverse=True)
    
    def _update_queue(self):
        pending = self.get_pending_tasks()
        self.task_queue = [t.id for t in pending if self._can_execute(t)]
    
    def _can_execute(self, task: Task) -> bool:
        for dep_id in task.dependencies:
            if dep_id in self.tasks and not self.tasks[dep_id].completed:
                return False
        return True

class SkillRegistry:
    def __init__(self):
        self.skills: Dict[str, Callable] = {}
    
    def register(self, name: str, skill_func: Callable):
        self.skills[name] = skill_func
    
    def get_skill(self, name: str) -> Optional[Callable]:
        return self.skills.get(name)
    
    def list_skills(self) -> List[str]:
        return list(self.skills.keys())

# FastAPI Application
app = FastAPI(title="RAG AI Agent API")

# Global agent instance
agent = RAGAIAgent(
    name="RayBot",
    personality={
        "traits": ["helpful", "analytical", "adaptive", "curious"],
        "learning_rate": 0.15,
        "curiosity": 0.9
    }
)

# Request/Response Models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    type: str = "general"

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI agent"""
    try:
        result = await agent.process_input(request.message)
        if isinstance(result, dict):
            return ChatResponse(**result)
        return ChatResponse(response=str(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the knowledge base"""
    if file.size > Config.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Save file temporarily
    temp_path = f"/tmp/{file.filename}"
    async with aiofiles.open(temp_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Ingest document
    result = await agent.ingest_document(temp_path)
    
    # Clean up
    os.remove(temp_path)
    
    if result["success"]:
        return JSONResponse(content=result)
    else:
        raise HTTPException(status_code=500, detail=result["message"])

@app.get("/status")
async def get_status():
    """Get agent status"""
    return agent.get_status()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": agent.name}

# Deployment script for small VM
if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs("/tmp", exist_ok=True)
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for small VM
        loop="asyncio",
        log_level="info"
    )