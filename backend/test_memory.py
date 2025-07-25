# Test script to debug memory issues
# Save this as test_memory.py and run it separately

import sqlite3
import json
import datetime
import uuid
import threading
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class Memory:
    id: str
    content: Dict[str, Any]
    timestamp: datetime.datetime
    category: str
    relevance_score: float = 1.0

class SimpleMemoryBank:
    def __init__(self, db_path: str = "test_memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        print(f"🔧 Creating memory database at: {self.db_path}")
        self._init_db()
    
    def _init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
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
            conn.close()
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"❌ Database init error: {e}")
            raise
    
    def store(self, content: Dict[str, Any], category: str) -> Memory:
        try:
            print(f"📝 Storing: {content} in category: {category}")
            
            memory = Memory(
                id=f"mem_{uuid.uuid4()}",
                content=content,
                timestamp=datetime.datetime.now(),
                category=category
            )
            
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?)",
                (memory.id, json.dumps(content), category, 
                 memory.timestamp.isoformat(), memory.relevance_score)
            )
            conn.commit()
            conn.close()
            
            print(f"✅ Stored memory: {memory.id}")
            return memory
            
        except Exception as e:
            print(f"❌ Storage error: {e}")
            raise
    
    def search(self, query: str, limit: int = 5) -> List[Memory]:
        try:
            print(f"🔍 Searching for: '{query}'")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT * FROM memories WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", limit)
            )
            
            memories = []
            for row in cursor:
                memory = Memory(
                    id=row[0],
                    content=json.loads(row[1]),
                    timestamp=datetime.datetime.fromisoformat(row[3]),
                    category=row[2],
                    relevance_score=row[4]
                )
                memories.append(memory)
            
            conn.close()
            print(f"✅ Found {len(memories)} memories")
            for mem in memories:
                print(f"   📄 {mem.category}: {mem.content}")
            
            return memories
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def count(self) -> int:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
            conn.close()
            print(f"📊 Total memories: {count}")
            return count
        except Exception as e:
            print(f"❌ Count error: {e}")
            return 0
    
    def list_all(self):
        """List all memories for debugging"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT * FROM memories ORDER BY timestamp DESC")
            
            print("\n📋 All memories in database:")
            for i, row in enumerate(cursor):
                print(f"   {i+1}. ID: {row[0]}")
                print(f"      Category: {row[2]}")
                print(f"      Content: {row[1]}")
                print(f"      Time: {row[3]}")
                print()
            
            conn.close()
        except Exception as e:
            print(f"❌ List error: {e}")

def test_memory_system():
    print("🧪 Testing Memory System...")
    print("=" * 50)
    
    # Create memory bank
    memory_bank = SimpleMemoryBank()
    
    # Test 1: Store Ray's name
    print("\n🧪 Test 1: Store Ray's name")
    memory_bank.store({"name": "Ray", "type": "user_name"}, "user_info")
    
    # Test 2: Store a preference
    print("\n🧪 Test 2: Store preference")
    memory_bank.store({"preference": "pizza", "user": "Ray"}, "user_preferences")
    
    # Test 3: Count memories
    print("\n🧪 Test 3: Count memories")
    total = memory_bank.count()
    
    # Test 4: Search for Ray
    print("\n🧪 Test 4: Search for Ray")
    ray_memories = memory_bank.search("Ray")
    
    # Test 5: Search for name
    print("\n🧪 Test 5: Search for 'name'")
    name_memories = memory_bank.search("name")
    
    # Test 6: List all memories
    print("\n🧪 Test 6: List all memories")
    memory_bank.list_all()
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"   Total memories stored: {total}")
    print(f"   Ray search results: {len(ray_memories)}")
    print(f"   Name search results: {len(name_memories)}")
    
    if total > 0 and len(ray_memories) > 0:
        print("✅ Memory system is working!")
        return True
    else:
        print("❌ Memory system has issues!")
        return False

if __name__ == "__main__":
    try:
        success = test_memory_system()
        if success:
            print("\n🎉 Memory test passed! Your memory system should work.")
        else:
            print("\n💥 Memory test failed! There are issues to fix.")
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()