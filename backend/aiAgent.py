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
import threading

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
from dotenv import load_dotenv
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path) 

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
    embedding: List[float] = None

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

class MemoryBank:
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        print(f"Initializing MemoryBank with database: {self.db_path}")
        try:
            self._init_db()
            print(f"MemoryBank initialized successfully. Current memory count: {self.count()}")
        except Exception as e:
            print(f"Error initializing MemoryBank: {e}")
            raise
    
    def _get_connection(self):
        """Get a new connection for each operation to avoid thread issues"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            return conn
        except Exception as e:
            print(f"Error creating database connection: {e}")
            raise
    
    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    category TEXT,
                    timestamp TEXT,
                    relevance_score REAL
                )
            """)
            conn.commit()
            print("Database tables created/verified")
    
    def store(self, content: Dict[str, Any], category: str) -> Memory:
        with self.lock:
            try:
                print(f"Attempting to store memory in category: {category}")
                print(f"Content: {content}")
                
                conn = self._get_connection()
                
                # Check current count and implement FIFO if needed
                cursor = conn.execute("SELECT COUNT(*) FROM memories")
                current_count = cursor.fetchone()[0]
                print(f"Current memory count: {current_count}")
                
                if current_count >= Config.MAX_MEMORY_ITEMS:
                    print("Memory limit reached, cleaning up old entries...")
                    conn.execute("""
                        DELETE FROM memories 
                        WHERE id IN (
                            SELECT id FROM memories 
                            ORDER BY timestamp ASC 
                            LIMIT 100
                        )
                    """)
                    conn.commit()
                    print("Old memories cleaned up")
                
                # Create new memory
                memory = Memory(
                    id=f"mem_{uuid.uuid4()}",
                    content=content,
                    timestamp=datetime.datetime.now(),
                    category=category
                )
                
                # Insert new memory
                conn.execute(
                    "INSERT INTO memories VALUES (?, ?, ?, ?, ?)",
                    (memory.id, json.dumps(content), category, 
                     memory.timestamp.isoformat(), memory.relevance_score)
                )
                conn.commit()
                conn.close()
                
                print(f"✓ Successfully stored memory: {memory.id}")
                return memory
                
            except Exception as e:
                print(f"✗ Error storing memory: {e}")
                print(f"Content that failed: {content}")
                if 'conn' in locals():
                    conn.close()
                raise
    
    def search(self, query: str, limit: int = 5) -> List[Memory]:
        with self.lock:
            try:
                print(f"Searching memories for: '{query}'")
                conn = self._get_connection()
                
                # Search in both content and category
                cursor = conn.execute(
                    """SELECT * FROM memories 
                    WHERE content LIKE ? OR category LIKE ?
                    ORDER BY timestamp DESC LIMIT ?""",
                    (f"%{query}%", f"%{query}%", limit)
                )
                
                memories = []
                for row in cursor:
                    try:
                        memory = Memory(
                            id=row[0],
                            content=json.loads(row[1]),
                            timestamp=datetime.datetime.fromisoformat(row[3]),
                            category=row[2],
                            relevance_score=row[4]
                        )
                        memories.append(memory)
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Error parsing memory {row[0]}: {e}")
                        continue
                
                conn.close()
                print(f"✓ Found {len(memories)} memories for query: '{query}'")
                for mem in memories:
                    print(f"  - {mem.category}: {mem.content}")
                return memories
                
            except Exception as e:
                print(f"✗ Error searching memories: {e}")
                if 'conn' in locals():
                    conn.close()
                return []
    
    def get_by_category(self, category: str, limit: int = 10) -> List[Memory]:
        """Get memories by category"""
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE category = ? ORDER BY timestamp DESC LIMIT ?",
                    (category, limit)
                )
                
                memories = []
                for row in cursor:
                    try:
                        memory = Memory(
                            id=row[0],
                            content=json.loads(row[1]),
                            timestamp=datetime.datetime.fromisoformat(row[3]),
                            category=row[2],
                            relevance_score=row[4]
                        )
                        memories.append(memory)
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Error parsing memory {row[0]}: {e}")
                        continue
                
                conn.close()
                return memories
                
            except Exception as e:
                print(f"Error getting memories by category: {e}")
                return []
    
    def count(self) -> int:
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("SELECT COUNT(*) FROM memories")
                count = cursor.fetchone()[0]
                conn.close()
                return count
            except Exception as e:
                print(f"Error counting memories: {e}")
                return 0

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
        self.memory_bank = MemoryBank("agent_memory.db")
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
            avg_rt_str = f"{avg_rt:.1f} ms"
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
        """Register built-in skills including fixed memory skills"""
        self.skill_registry.register("rag_query", self._rag_query)
        self.skill_registry.register("analyze_text", self._analyze_text)
        self.skill_registry.register("summarize", self._summarize_with_gemini)
        self.skill_registry.register("extract_entities", self._extract_entities)
        self.skill_registry.register("generate_response", self._generate_response)
        self.skill_registry.register("learn_pattern", self._learn_pattern)
        self.skill_registry.register("do_math", self._do_math)
        self.skill_registry.register("generate_code", self._generate_code_with_gemini)
        self.skill_registry.register("recall_fact", self._recall_fact)
    
    def debug_memory(self) -> Dict[str, Any]:
        """Debug memory contents"""
        try:
            total_memories = self.memory_bank.count()
            print(f"📊 Total memories in database: {total_memories}")
            
            # Get memories by category
            user_info_memories = self.memory_bank.get_by_category("user_info", limit=10)
            print(f"🔍 User info memories: {len(user_info_memories)}")
            
            # Search for specific terms
            name_memories = self.memory_bank.search("name", limit=10)
            print(f"🔍 Memories containing 'name': {len(name_memories)}")
            
            return {
                "total_memories": total_memories,
                "user_info_memories": len(user_info_memories),
                "user_info_details": [mem.content for mem in user_info_memories],
                "name_memories": len(name_memories),
                "name_details": [mem.content for mem in name_memories]
            }
        except Exception as e:
            print(f"❌ Error debugging memory: {e}")
            return {"error": str(e)}
    
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
    
    def _analyze_intent(self, text: str) -> Dict[str, Any]:
        """Enhanced intent analysis with memory operations"""
        lower = text.lower().strip()
        print(f"🔍 Analyzing intent for: '{text}'")
        
        # Memory storage detection - expanded patterns
        remember_patterns = [
            r"my name is\s+(\w+)",
            r"i am\s+(\w+)",
            r"call me\s+(\w+)",
            r"remember\s+(?:that\s+)?(.+)",
            r"store\s+(?:that\s+)?(.+)",
            r"save\s+(?:that\s+)?(.+)",
            r"learn\s+(?:that\s+)?(.+)"
        ]
        
        for pattern in remember_patterns:
            match = re.search(pattern, lower)
            if match:
                print("✅ Detected MEMORY STORE intent")
                return {
                    "type": "memory_store",
                    "action": "learn_pattern", 
                    "confidence": 0.95
                }
        
        # Memory recall detection - GREATLY EXPANDED patterns
        recall_patterns = [
            r"what'?s?\s+my\s+name",
            r"who\s+am\s+i",
            r"what\s+do\s+you\s+(?:know|remember)\s+about\s+me",
            r"do\s+you\s+remember\s+(.+)",
            r"recall\s+(.+)",
            r"what\s+did\s+i\s+(?:tell|say)\s+(?:you\s+)?about\s+(.+)",
            r"what'?s?\s+my\s+favorite\s+(.+)",
            r"what\s+is\s+my\s+(.+)",
            r"tell\s+me\s+(?:about\s+)?my\s+(.+)",
            r"do\s+you\s+know\s+my\s+(.+)",
            r"what\s+(?:do\s+you\s+)?(?:know\s+)?about\s+my\s+(.+)"
        ]
        
        for pattern in recall_patterns:
            if re.search(pattern, lower):
                print("✅ Detected MEMORY RECALL intent")
                return {
                    "type": "memory_recall",
                    "action": "recall_fact",
                    "confidence": 0.95
                }
        
        # Check for "my" keyword which often indicates personal info query
        if "my" in lower and any(q in lower for q in ["what", "tell", "know", "remember"]):
            print("✅ Detected MEMORY RECALL intent (my + question word)")
            return {
                "type": "memory_recall",
                "action": "recall_fact",
                "confidence": 0.9
            }
        
        # RAG detection
        if self._needs_rag(text, {}):
            print("✅ Detected RAG intent")
            return {
                "type": "rag",
                "action": "rag_query",
                "confidence": 0.95
            }
        
        # Math detection - be more specific to avoid conflicts
        math_patterns = [
            r"\d+\s*[\+\-\*/]\s*\d+",  # Basic arithmetic
            r"calculate\s+(.+)",
            r"what\s+is\s+\d+",
            r"math\s+problem"
        ]
        if any(re.search(pattern, lower) for pattern in math_patterns):
            print("✅ Detected MATH intent")
            return {
                "type": "math",
                "action": "do_math",
                "confidence": 0.9
            }
        
        # Default to general response
        print("✅ Detected GENERAL intent")
        return {
            "type": "general",
            "action": "generate_response",
            "confidence": 0.8
        }
    
    def _execute_skill(self, intent: Dict[str, Any], input_text: str) -> Any:
        """Execute skill based on intent"""
        skill_name = intent["action"]
        print(f"🚀 Executing skill: {skill_name}")
        print(f"🚀 Available skills: {self.skill_registry.list_skills()}")
        
        skill = self.skill_registry.get_skill(skill_name)
        
        if skill:
            print(f"✅ Found skill: {skill_name}")
            try:
                # Handle async skills
                if asyncio.iscoroutinefunction(skill):
                    print(f"🔄 Skill {skill_name} is async")
                    return asyncio.create_task(skill(input_text))
                else:
                    print(f"🔄 Skill {skill_name} is sync")
                    result = skill(input_text)
                    print(f"✅ Skill result: {result}")
                    return result
            except Exception as e:
                print(f"❌ Error executing skill {skill_name}: {e}")
                traceback.print_exc()
                return {"response": f"Error executing {skill_name}: {str(e)}", "type": "error"}
        else:
            print(f"❌ Skill not found: {skill_name}")
            return {"response": "I'm not sure how to handle that request yet, but I'm always learning!", "type": "general"}
    
    def _analyze_text(self, text: str) -> Dict[str, str]:
        word_count = len(text.split())
        sentences = text.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Use Gemini for deeper analysis
        gemini_analysis = self.gemini.generate_response(
            f"Analyze the sentiment and key themes in this text: {text[:500]}"
        )
        
        response = (
            f"Here's what I found about your text!\n"
            f"- Word count: {word_count}\n"
            f"- Sentences: {sentence_count}\n"
            f"- AI Analysis: {gemini_analysis}\n"
        )
        
        return {"response": response, "type": "analysis"}
    
    def _extract_entities(self, text: str) -> Dict[str, str]:
        # Use Gemini for better entity extraction
        prompt = f"Extract all named entities (people, places, organizations, dates, numbers) from this text: {text}"
        response = self.gemini.generate_response(prompt)
        return {"response": response, "type": "entities"}
    
    def _generate_response(self, text: str) -> Dict[str, str]:
        """Use Gemini for general responses"""
        try:
            response = self.gemini.generate_response(text)
            return {
                "response": response,
                "type": "general"
            }
        except Exception as e:
            return {
                "response": f"I encountered an error: {str(e)}",
                "type": "error"
            }
    
    def _learn_pattern(self, text: str) -> Dict[str, str]:
        """Store information in memory - IMPROVED VERSION"""
        self.state = AgentState.LEARNING
        print(f"🧠 Learning pattern: {text}")
        
        try:
            # Extract name patterns
            name_patterns = [
                r"my name is\s+(\w+)",
                r"i am\s+(\w+)",
                r"call me\s+(\w+)"
            ]
            
            name = None
            for pattern in name_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    name = match.group(1).title()
                    break
            
            if name:
                # Store name information
                memory_entry = {
                    "name": name,
                    "full_text": text,
                    "learned_at": datetime.datetime.now().isoformat(),
                    "type": "user_name"
                }
                category = "user_info"
                
                # Store multiple ways for better recall
                stored_memory = self.memory_bank.store(
                    content=memory_entry,
                    category=category
                )
                
                # Also store a simplified version
                self.memory_bank.store(
                    content={"user_name": name, "timestamp": datetime.datetime.now().isoformat()},
                    category="name"
                )
                
                response_msg = f"✓ Nice to meet you, {name}! I'll remember your name."
            else:
                # General learning
                memory_entry = {
                    "content": text,
                    "learned_at": datetime.datetime.now().isoformat(),
                    "type": "general_fact"
                }
                category = "learned_pattern"
                
                stored_memory = self.memory_bank.store(
                    content=memory_entry,
                    category=category
                )
                
                response_msg = f"✓ I've learned that: '{text}'"
            
            # Also add to vector store for RAG
            self.vector_store.add_documents([{
                "id": str(uuid.uuid4()),
                "content": text,
                "metadata": {
                    "type": "learned_pattern",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "category": category
                }
            }])
            
            print(f"✅ Memory stored successfully with ID: {stored_memory.id}")
            self.experience_points += 10
            self.state = AgentState.IDLE
            
            return {
                "response": response_msg,
                "type": "memory_stored"
            }
            
        except Exception as e:
            print(f"❌ Error in _learn_pattern: {e}")
            traceback.print_exc()
            self.state = AgentState.IDLE
            return {
                "response": f"✗ Sorry, I had trouble storing that memory: {str(e)}",
                "type": "error"
            }
    
    def _recall_fact(self, text: str) -> Dict[str, str]:
        """Enhanced recall that searches memory effectively"""
        print(f"🔍 Recalling fact for: '{text}'")
        
        try:
            # Check total memory count
            total = self.memory_bank.count()
            print(f"📊 Total memories available: {total}")
            
            # For name queries, check user_info category first
            if "name" in text.lower():
                print("🔍 Looking for name in user_info category...")
                
                # Search user_info category
                user_memories = self.memory_bank.get_by_category("user_info", limit=10)
                print(f"Found {len(user_memories)} user_info memories")
                
                # Look for most recent name
                for memory in user_memories:
                    if "name" in memory.content:
                        name = memory.content["name"]
                        print(f"✅ Found name: {name}")
                        return {
                            "response": f"Your name is {name}! I remember you told me that.",
                            "type": "memory_recall"
                        }
                
                # Also check simplified name category
                name_memories = self.memory_bank.get_by_category("name", limit=5)
                for memory in name_memories:
                    if "user_name" in memory.content:
                        name = memory.content["user_name"]
                        return {
                            "response": f"Your name is {name}!",
                            "type": "memory_recall"
                        }
            
            # Extract what the user is asking about
            query_patterns = [
                r"what'?s?\s+my\s+favorite\s+(.+)",
                r"what\s+is\s+my\s+(.+)",
                r"tell\s+me\s+(?:about\s+)?my\s+(.+)",
                r"do\s+you\s+know\s+my\s+(.+)",
                r"what\s+(?:do\s+you\s+)?(?:know\s+)?about\s+my\s+(.+)"
            ]
            
            search_topic = None
            for pattern in query_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    search_topic = match.group(1).strip()
                    print(f"🔍 User is asking about their: {search_topic}")
                    break
            
            # Search for the specific topic
            if search_topic:
                # Search for memories containing the topic
                topic_memories = self.memory_bank.search(search_topic, limit=10)
                print(f"🔍 Found {len(topic_memories)} memories about '{search_topic}'")
                
                for memory in topic_memories:
                    content = memory.content
                    
                    # Check if this memory contains info about the topic
                    if isinstance(content, dict):
                        # Look in the stored text
                        stored_text = content.get("content", content.get("full_text", content.get("text", "")))
                        if search_topic in stored_text.lower():
                            # Extract the relevant part
                            if "favorite" in text.lower() and "favorite" in stored_text.lower():
                                # Try to extract "favorite X is Y" pattern
                                fav_match = re.search(rf"(?:my\s+)?favorite\s+{search_topic}\s+is\s+(\w+)", stored_text.lower())
                                if fav_match:
                                    item = fav_match.group(1)
                                    return {
                                        "response": f"Your favorite {search_topic} is {item}!",
                                        "type": "memory_recall"
                                    }
                            
                            # Generic response with the stored content
                            return {
                                "response": f"I remember you told me: {stored_text}",
                                "type": "memory_recall"
                            }
            
            # General search - extract keywords and search
            keywords = re.findall(r'\b\w{3,}\b', text.lower())  # Words with 3+ chars
            # Filter out common words
            stop_words = {"what", "whats", "tell", "know", "about", "remember", "you", "the"}
            keywords = [k for k in keywords if k not in stop_words]
            print(f"🔍 Searching for keywords: {keywords}")
            
            for keyword in keywords[:5]:  # Try first 5 keywords
                memories = self.memory_bank.search(keyword, limit=5)
                
                if memories:
                    print(f"✅ Found {len(memories)} memories for '{keyword}'")
                    memory = memories[0]  # Use most recent
                    
                    # Format response based on memory type
                    if isinstance(memory.content, dict):
                        if "content" in memory.content:
                            recalled_text = memory.content["content"]
                        elif "full_text" in memory.content:
                            recalled_text = memory.content["full_text"]
                        elif "text" in memory.content:
                            recalled_text = memory.content["text"]
                        else:
                            recalled_text = str(memory.content)
                        
                        return {
                            "response": f"I remember: {recalled_text}",
                            "type": "memory_recall"
                        }
            
            # Try vector store as fallback
            vector_results = self.vector_store.search(text, k=1)
            if vector_results:
                return {
                    "response": f"From my knowledge base: {vector_results[0]['content']}",
                    "type": "memory_recall"
                }
            
            # No memories found
            print("❌ No relevant memories found")
            return {
                "response": f"I don't have any specific memories about that yet. I currently have {total} memories stored. Try teaching me something first!",
                "type": "memory_recall"
            }
            
        except Exception as e:
            print(f"❌ Error in _recall_fact: {e}")
            traceback.print_exc()
            return {
                "response": f"Sorry, I had trouble searching my memory: {str(e)}",
                "type": "error"
            }
    
    def _do_math(self, text: str) -> Dict[str, str]:
        # First try local calculation
        try:
            # Simple math expression parsing
            expr = text.lower().replace('what is', '').replace('calculate', '').strip()
            # Remove question marks and other punctuation
            expr = re.sub(r'[^\d+\-*/().\s]', '', expr)
            
            if expr:
                result = eval(expr)
                return {
                    "response": f"The answer is: {result}",
                    "type": "math"
                }
        except:
            pass
        
        # Fall back to Gemini for complex math
        try:
            response = self.gemini.generate_response(f"Solve this math problem: {text}")
            return {
                "response": response,
                "type": "math"
            }
        except Exception as e:
            return {
                "response": f"I had trouble with that math problem: {str(e)}",
                "type": "error"
            }

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

@app.get("/api/debug/memory")
async def debug_memory():
    """Debug memory contents"""
    try:
        return agent.debug_memory()
    except Exception as e:
        return {"error": str(e)}

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