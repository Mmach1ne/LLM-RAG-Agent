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

# Agent States
class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    EXECUTING = "executing"

# Task Priority Levels
class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """Represents a task for the agent to execute"""
    id: str
    description: str
    priority: Priority
    created_at: datetime.datetime
    completed: bool = False
    result: Optional[Any] = None
    dependencies: List[str] = field(default_factory=list)

@dataclass
class Memory:
    """Represents a memory entry"""
    id: str
    content: Dict[str, Any]
    timestamp: datetime.datetime
    category: str
    relevance_score: float = 1.0

class AIAgent:
    """Advanced AI Agent with multiple capabilities"""
    
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
        
        # Initialize default skills
        self._register_default_skills()
        
        # Learning parameters
        self.experience_points = 0
        self.skill_levels = defaultdict(int)
        
    def _register_default_skills(self):
        """Register built-in skills"""
        self.skill_registry.register("analyze_text", self._analyze_text)
        self.skill_registry.register("summarize", self._summarize)
        self.skill_registry.register("extract_entities", self._extract_entities)
        self.skill_registry.register("generate_response", self._generate_response)
        self.skill_registry.register("learn_pattern", self._learn_pattern)
        
    def process_input(self, user_input: str) -> str:
        """Process user input and generate appropriate response"""
        self.state = AgentState.PROCESSING
        
        # Store conversation
        self.conversation_history.append({
            "timestamp": datetime.datetime.now(),
            "user": user_input,
            "agent": None
        })
        
        # Analyze intent
        intent = self._analyze_intent(user_input)
        
        # Create task based on intent
        task = self.task_manager.create_task(
            description=f"Process: {intent['action']}",
            priority=Priority.MEDIUM
        )
        
        # Execute appropriate skill
        response = self._execute_skill(intent, user_input)
        
        # Update conversation history
        self.conversation_history[-1]["agent"] = response
        
        # Learn from interaction
        self._learn_from_interaction(user_input, response, intent)
        
        # Complete task
        self.task_manager.complete_task(task.id, response)
        
        self.state = AgentState.IDLE
        return response
    
    def _analyze_intent(self, text: str) -> Dict[str, Any]:
        """Analyze user intent from input text"""
        # Simple pattern matching for demonstration
        patterns = {
            "question": r"\?$|^(what|who|where|when|why|how)",
            "command": r"^(do|make|create|build|analyze|summarize)",
            "statement": r"^(i|the|it|this|that)",
        }
        
        intent_type = "unknown"
        for intent, pattern in patterns.items():
            if re.search(pattern, text.lower()):
                intent_type = intent
                break
        
        # Determine action based on intent
        action_map = {
            "question": "generate_response",
            "command": "analyze_text",
            "statement": "learn_pattern"
        }
        
        return {
            "type": intent_type,
            "action": action_map.get(intent_type, "generate_response"),
            "confidence": 0.8
        }
    
    def _execute_skill(self, intent: Dict[str, Any], input_text: str) -> str:
        """Execute skill based on intent"""
        skill_name = intent["action"]
        skill = self.skill_registry.get_skill(skill_name)
        
        if skill:
            return skill(input_text)
        else:
            return "I'm not sure how to handle that request yet."
    
    def _analyze_text(self, text: str) -> str:
        """Analyze text and provide insights"""
        word_count = len(text.split())
        sentences = text.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor']
        
        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        sentiment = "neutral"
        if positive_score > negative_score:
            sentiment = "positive"
        elif negative_score > positive_score:
            sentiment = "negative"
        
        analysis = f"""Text Analysis:
- Word count: {word_count}
- Sentence count: {sentence_count}
- Average words per sentence: {word_count / max(sentence_count, 1):.1f}
- Sentiment: {sentiment}
- Complexity: {'simple' if word_count < 50 else 'moderate' if word_count < 150 else 'complex'}"""
        
        return analysis
    
    def _summarize(self, text: str) -> str:
        """Generate a summary of the text"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 2:
            return text
        
        # Simple extractive summarization
        # In practice, you'd use more sophisticated methods
        key_sentences = sentences[:2]  # Take first two sentences
        
        return "Summary: " + '. '.join(key_sentences) + "."
    
    def _extract_entities(self, text: str) -> str:
        """Extract named entities from text"""
        # Simple pattern-based extraction
        # In practice, you'd use NLP libraries like spaCy
        
        # Look for capitalized words (potential proper nouns)
        words = text.split()
        entities = [word for word in words if word[0].isupper() and len(word) > 1]
        
        # Look for numbers
        numbers = re.findall(r'\b\d+\b', text)
        
        # Look for dates (simple pattern)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        
        result = "Extracted Entities:\n"
        if entities:
            result += f"- Names/Places: {', '.join(set(entities))}\n"
        if numbers:
            result += f"- Numbers: {', '.join(numbers)}\n"
        if dates:
            result += f"- Dates: {', '.join(dates)}\n"
        
        return result if any([entities, numbers, dates]) else "No specific entities found."
    
    def _generate_response(self, text: str) -> str:
        """Generate contextual response"""
        # Check memory for relevant information
        relevant_memories = self.memory_bank.search(text, limit=3)
        
        # Build context from memories
        context = ""
        if relevant_memories:
            context = "Based on my memory: " + "; ".join([m.content.get("info", "") for m in relevant_memories])
        
        # Generate response based on personality and context
        responses = {
            "helpful": [
                "I'd be happy to help with that.",
                "Let me assist you with this.",
                "Here's what I can tell you:"
            ],
            "analytical": [
                "Let me analyze this for you.",
                "Based on my analysis:",
                "Here's my assessment:"
            ],
            "adaptive": [
                "I'm learning from our interaction.",
                "I'll remember this for next time.",
                "I'm adapting to better serve you."
            ]
        }
        
        # Select response based on primary trait
        primary_trait = self.personality["traits"][0]
        response_prefix = random.choice(responses.get(primary_trait, ["I understand."]))
        
        # Simple response generation
        if "?" in text:
            response = f"{response_prefix} That's an interesting question. {context}"
        else:
            response = f"{response_prefix} I've noted your input. {context}"
        
        return response
    
    def _learn_pattern(self, text: str) -> str:
        """Learn from patterns in text"""
        self.state = AgentState.LEARNING
        
        # Extract patterns (simplified)
        words = text.lower().split()
        
        # Store in memory
        memory_entry = {
            "text": text,
            "words": words,
            "length": len(words),
            "info": f"Learned pattern with {len(words)} words"
        }
        
        self.memory_bank.store(
            content=memory_entry,
            category="pattern"
        )
        
        # Increase experience
        self.experience_points += 10
        self.skill_levels["pattern_recognition"] += 1
        
        self.state = AgentState.IDLE
        return f"I've learned from this pattern. My pattern recognition skill is now level {self.skill_levels['pattern_recognition']}."
    
    def _learn_from_interaction(self, user_input: str, response: str, intent: Dict[str, Any]):
        """Learn from the interaction"""
        # Store interaction in memory
        self.memory_bank.store(
            content={
                "user_input": user_input,
                "response": response,
                "intent": intent,
                "success": True  # In practice, you'd measure this
            },
            category="interaction"
        )
        
        # Update skill levels
        skill_used = intent["action"]
        self.skill_levels[skill_used] += 1
        self.experience_points += 5
    
    def add_goal(self, goal: str, priority: Priority = Priority.MEDIUM):
        """Add a goal for the agent to work towards"""
        task = self.task_manager.create_task(
            description=f"Goal: {goal}",
            priority=priority
        )
        self.goals.append(task)
        return f"Goal added: {goal}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "experience": self.experience_points,
            "skills": dict(self.skill_levels),
            "active_tasks": len(self.task_manager.get_pending_tasks()),
            "memories": self.memory_bank.count(),
            "goals": len(self.goals),
            "personality": self.personality
        }

class MemoryBank:
    """Manages agent's memory storage and retrieval"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize memory database"""
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
        """Store a new memory"""
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
        
        return memory
    
    def search(self, query: str, limit: int = 5) -> List[Memory]:
        """Search memories by relevance"""
        # Simple search - in practice, use vector similarity
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
        """Get total memory count"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]

class TaskManager:
    """Manages agent tasks and execution"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
    
    def create_task(self, description: str, priority: Priority = Priority.MEDIUM, 
                   dependencies: List[str] = None) -> Task:
        """Create a new task"""
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
        """Mark task as completed"""
        if task_id in self.tasks:
            self.tasks[task_id].completed = True
            self.tasks[task_id].result = result
            self._update_queue()
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks sorted by priority"""
        pending = [t for t in self.tasks.values() if not t.completed]
        return sorted(pending, key=lambda t: t.priority.value, reverse=True)
    
    def _update_queue(self):
        """Update task execution queue"""
        pending = self.get_pending_tasks()
        self.task_queue = [t.id for t in pending if self._can_execute(t)]
    
    def _can_execute(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id in self.tasks and not self.tasks[dep_id].completed:
                return False
        return True

class SkillRegistry:
    """Registry for agent skills"""
    
    def __init__(self):
        self.skills: Dict[str, Callable] = {}
    
    def register(self, name: str, skill_func: Callable):
        """Register a new skill"""
        self.skills[name] = skill_func
    
    def get_skill(self, name: str) -> Optional[Callable]:
        """Get skill by name"""
        return self.skills.get(name)
    
    def list_skills(self) -> List[str]:
        """List all available skills"""
        return list(self.skills.keys())

# Example usage and demonstration
if __name__ == "__main__":
    # Create an AI agent
    agent = AIAgent(
        name="Atlas",
        personality={
            "traits": ["helpful", "analytical", "adaptive", "curious"],
            "learning_rate": 0.15,
            "curiosity": 0.9
        }
    )
    
    # Add some goals
    print(agent.add_goal("Learn from user interactions", Priority.HIGH))
    print(agent.add_goal("Improve response quality", Priority.MEDIUM))
    
    # Demonstrate various capabilities
    print("\n=== AI Agent Demonstration ===\n")
    
    # Test 1: Question answering
    response1 = agent.process_input("What can you help me with?")
    print(f"User: What can you help me with?")
    print(f"Agent: {response1}\n")
    
    # Test 2: Text analysis
    response2 = agent.process_input("Analyze this text: The quick brown fox jumps over the lazy dog. It's a beautiful day today!")
    print(f"User: Analyze this text: The quick brown fox jumps over the lazy dog. It's a beautiful day today!")
    print(f"Agent: {response2}\n")
    
    # Test 3: Pattern learning
    response3 = agent.process_input("Remember that I prefer detailed explanations")
    print(f"User: Remember that I prefer detailed explanations")
    print(f"Agent: {response3}\n")
    
    # Test 4: Entity extraction
    response4 = agent.process_input("Extract entities from: John Smith visited Paris on 12/25/2023 and spent $500")
    print(f"User: Extract entities from: John Smith visited Paris on 12/25/2023 and spent $500")
    print(f"Agent: {response4}\n")
    
    # Show agent status
    print("\n=== Agent Status ===")
    status = agent.get_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    # Custom skill example
    def custom_skill(text: str) -> str:
        """Example of adding a custom skill"""
        return f"Custom skill processed: {text.upper()}"
    
    # Register custom skill
    agent.skill_registry.register("custom_process", custom_skill)
    print(f"\nRegistered skills: {agent.skill_registry.list_skills()}")