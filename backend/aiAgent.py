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
    "Represents a task for the agent to execute"
    id: str
    description: str
    priority: Priority
    created_at: datetime.datetime
    completed: bool = False
    result: Optional[Any] = None
    dependencies: List[str] = field(default_factory=list)

@dataclass
class Memory:
    "Represents a memory entry"
    id: str
    content: Dict[str, Any]
    timestamp: datetime.datetime
    category: str
    relevance_score: float = 1.0

class AIAgent:
    
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
        self.skill_registry.register("do_math", self._do_math)
        self.skill_registry.register("generate_code", self._generate_code)
        self.skill_registry.register("recall_fact", self._recall_fact)
        self.skill_registry.register("weather", self._weather_widget)

    def process_input(self, user_input: str) -> Any:
        self.state = AgentState.PROCESSING
        task = self.task_manager.create_task(f"Process: {user_input}", Priority.MEDIUM)
        time.sleep(1)  # Simulate processing
        self.state = AgentState.EXECUTING
        time.sleep(1)  # Simulate execution
        intent = self._analyze_intent(user_input)
        response = self._execute_skill(intent, user_input)
        self.task_manager.complete_task(task.id, response)
        self.state = AgentState.IDLE
        return response
    
    def _analyze_intent(self, text: str) -> Dict[str, Any]:
        # Weather detection
        weather_keywords = ["weather", "temperature", "forecast", "rain", "sunny", "cloudy", "windy", "snow", "humidity"]
        if any(word in text.lower() for word in weather_keywords):
            return {
                "type": "weather",
                "action": "weather",
                "confidence": 0.98
            }
        # 1) Code detection - broad patterns for coding requests
        code_keywords = [
            "write a function", "write me a function", "write function", "python code", "show me code", "show code", "implement", "define a function", "create a function", "how do i code", "generate python", "function for", "code for", "print", "output", "count to", "count from",
            "write code", "create code", "generate code", "make a function", "make function", "build a function", "build function", "code example", "python example", "programming", "script", "algorithm", "function that", "class", "method", "loop", "if statement", "while loop", "for loop",
            "sort", "filter", "map", "list comprehension", "dictionary", "tuple", "set", "file handling", "read file", "write file", "database", "api", "web scraping", "data processing", "machine learning", "data analysis", "visualization", "plot", "chart", "graph"
        ]
        # Check for keyword matches
        if any(word in text.lower() for word in code_keywords):
            return {
                "type": "code",
                "action": "generate_code",
                "confidence": 0.95
            }
        
        # Check for regex patterns for more complex coding requests
        code_patterns = [
            r'write\s+(?:me\s+)?(?:a\s+)?(?:python\s+)?(?:function|code|script|program)',
            r'create\s+(?:me\s+)?(?:a\s+)?(?:python\s+)?(?:function|code|script|program)',
            r'generate\s+(?:me\s+)?(?:a\s+)?(?:python\s+)?(?:function|code|script|program)',
            r'build\s+(?:me\s+)?(?:a\s+)?(?:python\s+)?(?:function|code|script|program)',
            r'make\s+(?:me\s+)?(?:a\s+)?(?:python\s+)?(?:function|code|script|program)',
            r'how\s+(?:do\s+i\s+)?(?:write|create|generate|build|make)\s+(?:a\s+)?(?:python\s+)?(?:function|code|script|program)',
            r'can\s+you\s+(?:write|create|generate|build|make)\s+(?:me\s+)?(?:a\s+)?(?:python\s+)?(?:function|code|script|program)',
            r'i\s+need\s+(?:a\s+)?(?:python\s+)?(?:function|code|script|program)',
            r'help\s+me\s+(?:write|create|generate|build|make)\s+(?:a\s+)?(?:python\s+)?(?:function|code|script|program)',
            r'example\s+(?:of\s+)?(?:python\s+)?(?:function|code|script|program)',
            r'sample\s+(?:of\s+)?(?:python\s+)?(?:function|code|script|program)',
            r'python\s+(?:function|code|script|program)\s+(?:for|that|to)',
            r'function\s+(?:that|to|for)\s+',
            r'class\s+(?:that|to|for)\s+',
            r'method\s+(?:that|to|for)\s+',
            r'algorithm\s+(?:for|to|that)\s+',
            r'loop\s+(?:that|to|for)\s+',
            r'if\s+statement\s+(?:for|to|that)\s+',
            r'while\s+loop\s+(?:for|to|that)\s+',
            r'for\s+loop\s+(?:for|to|that)\s+'
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text.lower()):
                return {
                    "type": "code",
                    "action": "generate_code",
                    "confidence": 0.95
                }
        # 2) Recall detection …
        recall_patterns = [
            r'where do i (study|live|work)',
            r'what is my (name|school|university|job|major|hobby|favorite|favourite)',
            r'what did i tell you',
            r'do you remember (.+)',
            r'who am i',
        ]
        for pat in recall_patterns:
            if re.search(pat, text.lower()):
                return {
                    "type": "recall",
                    "action": "recall_fact",
                    "confidence": 0.95
                }
        # 3) Learning trigger: look for “remember that …” or “remember my …”
        lower = text.lower().strip()
        if lower.startswith("remember ") or lower.startswith("remember that ") or lower.startswith("remember my "):
            return {
                "type": "learning",
                "action": "learn_pattern",
                "confidence": 0.95
            }
        # 4) Math detection …
        math_keywords = ["add", "subtract", "multiply", "divide", "plus", "minus", "times", "over", "math", "calculate", "solve", "what is", "^", "sqrt", "log", "sin", "cos", "tan", "exp", "power", "root"]
        if any(word in text.lower() for word in math_keywords) or re.search(r"\d+\s*([+\-*/^])\s*\d+", text):
            return {
                "type": "math",
                "action": "do_math",
                "confidence": 0.95
            }
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
    
    def _execute_skill(self, intent: Dict[str, Any], input_text: str) -> Any:
        """Execute skill based on intent"""
        skill_name = intent["action"]
        skill = self.skill_registry.get_skill(skill_name)
        
        if skill:
            return skill(input_text)
        else:
            return {"response": "I'm not sure how to handle that request yet, but I'm always learning! If you want to analyze, summarize, or extract info, just let me know."}
    
    def _analyze_text(self, text: str) -> str:
        word_count = len(text.split())
        sentences = text.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
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
        return (
            f"Here's what I found about your text!\n"
            f"- Word count: {word_count}\n"
            f"- Sentences: {sentence_count}\n"
            f"- Sentiment: {sentiment}\n"
            f"- Complexity: {'simple' if word_count < 50 else 'moderate' if word_count < 150 else 'complex'}\n"
            f"If you'd like a summary or want me to extract details, just let me know!"
        )

    def _summarize(self, text: str) -> str:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 2:
            summary = text
        else:
            key_sentences = sentences[:2]
            summary = '. '.join(key_sentences) + "."
        return f"Here's a quick summary for you: {summary} If you want more details, just ask!"

    def _extract_entities(self, text: str) -> str:
        words = text.split()
        entities = [word for word in words if word[0].isupper() and len(word) > 1]
        numbers = re.findall(r'\b\d+\b', text)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        if not any([entities, numbers, dates]):
            return "I couldn't find any specific names, numbers, or dates in your message. If you want me to look for something else, just let me know!"
        result = "Here's what I found in your message:\n"
        if entities:
            result += f"- Names/Places: {', '.join(set(entities))}\n"
        if numbers:
            result += f"- Numbers: {', '.join(numbers)}\n"
        if dates:
            result += f"- Dates: {', '.join(dates)}\n"
        result += "Let me know if you want more details about any of these!"
        return result

    def _generate_response(self, text: str) -> str:
        # Friendly small talk and fallback
        text_lower = text.lower()
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if any(greet in text_lower for greet in greetings):
            return "Hello! 😊 How can I help you today?"
        if "how are you" in text_lower:
            return "I'm just a bunch of code, but I'm always happy to chat and help you!"
        if "thank" in text_lower:
            return "You're very welcome! If you have more questions, just ask."
        if "who are you" in text_lower or "what are you" in text_lower:
            return "I'm Atlas, your friendly AI assistant! I'm here to help with anything you need."
        if "help" in text_lower:
            return "Of course! You can ask me to analyze text, summarize, extract information, or just chat. What would you like to do?"
        # Fallback for general chat
        return f"I'm not sure how to help with that yet, but I'm always learning! If you want to analyze, summarize, or extract info, just let me know."
    
    def _learn_pattern(self, text: str) -> str:
        self.state = AgentState.LEARNING
        words = text.lower().split()
        import re
        lower_text = text.lower().strip()
        is_fact = False
        cleaned_text = re.sub(r'^(remember( that| my)?\s*)', '', text, flags=re.IGNORECASE).strip()
        if 'remember' in lower_text and (re.search(r'\bi\b', lower_text) or re.search(r'\bmy\b', lower_text)):
            is_fact = True
        else:
            fact_patterns = [
                r'^i am ', r'^i study ', r'^i work ', r'^i live ', r'^my name is ', r'^my school is ', r'^my university is ', r'^my job is ', r'^my major is ', r'^my hobby is ', r'^my favorite', r'^my favourite',
                r'^remember that i ', r'^remember i ', r'^remember my ', r'^remember that my '
            ]
            is_fact = any(re.search(pat, lower_text) for pat in fact_patterns)
        memory_entry = {
            "text": cleaned_text,
            "words": cleaned_text.lower().split(),
            "length": len(cleaned_text.split()),
            "info": f"Learned pattern with {len(cleaned_text.split())} words"
        }
        category = "user_fact" if is_fact else "pattern"
        self.memory_bank.store(
            content=memory_entry,
            category=category
        )
        self.experience_points += 10
        self.skill_levels["pattern_recognition"] += 1
        self.state = AgentState.IDLE
        if is_fact:
            return f"Thanks for sharing! I'll remember that: '{cleaned_text}' as a fact about you."
        else:
            return f"Thanks for sharing! I'll remember that: '{cleaned_text}'"
    
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

    def _correct_typos(self, text: str) -> str:
        # Common math keywords/functions and their correct forms
        math_terms = [
            'square', 'cube', 'root', 'sqrt', 'log', 'ln', 'sin', 'cos', 'tan', 'exp', 'power', 'percent',
            'plus', 'minus', 'times', 'over', 'divided', 'multiplied', 'calculate', 'solve', 'what', 'is',
            'of', 'by', 'to', 'the', 'and', 'for', 'abs', 'round', 'pow', 'min', 'max', 'radians', 'degrees'
        ]
        # Common number words
        number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'for': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10', 'eleven': '11', 'twelve': '12',
            'thirteen': '13', 'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20'
        }
        typo_dict = {
            'sqaure': 'square', 'squre': 'square', 'squar': 'square', 'sqare': 'square',
            'sqaured': 'squared', 'squred': 'squared', 'cubed': 'cubed', 'cubic': 'cube',
            'rooot': 'root', 'squroot': 'square root', 'squareroot': 'square root',
            'sqaure root': 'square root', 'squarer oot': 'square root',
            'sine': 'sin', 'cosine': 'cos', 'tanget': 'tan', 'tangent': 'tan',
            'loog': 'log', 'lgo': 'log', 'lnn': 'ln', 'expnential': 'exp',
            'devide': 'divide', 'multipliedby': 'multiplied by', 'devided': 'divided',
            'devide by': 'divided by', 'pluss': 'plus', 'minuss': 'minus', 'timess': 'times',
            'precent': 'percent', 'precentage': 'percent', 'precent of': 'percent of',
            'sqaure of': 'square of', 'cubed of': 'cube of',
        }
        # Replace typo_dict first
        for typo, correct in typo_dict.items():
            text = re.sub(r'\b' + re.escape(typo) + r'\b', correct, text)
        # Fuzzy match for math terms
        words = text.split()
        for i, word in enumerate(words):
            if word not in math_terms:
                close = difflib.get_close_matches(word, math_terms, n=1, cutoff=0.85)
                if close:
                    words[i] = close[0]
            # Fuzzy match for number words
            if word not in number_words.values() and word in number_words:
                words[i] = number_words[word]
            elif word not in number_words.values():
                close_num = difflib.get_close_matches(word, list(number_words.keys()), n=1, cutoff=0.85)
                if close_num:
                    words[i] = number_words[close_num[0]]
        return ' '.join(words)

    def _do_math(self, text: str) -> str:
        """Attempt to solve math expressions, including advanced functions and natural language phrases, with typo correction."""
        intro = [
            "Let's crunch those numbers!",
            "Here's the math result:",
            "Math to the rescue!",
            "Here's what I calculated:",
            "Numbers are my friends!"
        ]
        # Correct typos first
        text = self._correct_typos(text)
        expr = text.lower().replace('what is', '').replace('calculate', '').replace('solve', '').replace('?', '').strip()
        expr = expr.replace('^', '**')
        expr = expr.replace('plus', '+').replace('minus', '-').replace('times', '*').replace('over', '/').replace('divided by', '/').replace('multiplied by', '*')
        expr = re.sub(r'\b(the|of|by|a|an|and|to|for)\b', '', expr)

        # Natural language math phrase replacements
        expr = re.sub(r'square root\s*(?:of)?\s*(\d+(?:\.\d+)?)', r'sqrt(\1)', expr)
        expr = re.sub(r'cube root\s*(?:of)?\s*(\d+(?:\.\d+)?)', r'pow(\1, 1/3)', expr)
        expr = re.sub(r'log\s*base\s*(\d+(?:\.\d+)?)\s*(?:of)?\s*(\d+(?:\.\d+)?)', r'log(\2, \1)', expr)
        expr = re.sub(r'(\d+(?:\.\d+)?)\s*to the power of\s*(\d+(?:\.\d+)?)', r'pow(\1, \2)', expr)
        expr = re.sub(r'(\d+(?:\.\d+)?)\s*squared', r'pow(\1, 2)', expr)
        expr = re.sub(r'(\d+(?:\.\d+)?)\s*cubed', r'pow(\1, 3)', expr)
        expr = re.sub(r'sin\s*\(?\s*(\d+(?:\.\d+)?)\s*degrees?\)?', r'sin(radians(\1))', expr)
        expr = re.sub(r'cos\s*\(?\s*(\d+(?:\.\d+)?)\s*degrees?\)?', r'cos(radians(\1))', expr)
        expr = re.sub(r'tan\s*\(?\s*(\d+(?:\.\d+)?)\s*degrees?\)?', r'tan(radians(\1))', expr)
        expr = re.sub(r'sin\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?', r'sin(\1)', expr)
        expr = re.sub(r'cos\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?', r'cos(\1)', expr)
        expr = re.sub(r'tan\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?', r'tan(\1)', expr)
        expr = re.sub(r'exp\s*(?:of)?\s*(\d+(?:\.\d+)?)', r'exp(\1)', expr)
        expr = re.sub(r'ln\s*(?:of)?\s*(\d+(?:\.\d+)?)', r'log(\1)', expr)
        expr = re.sub(r'log\s*(?:of)?\s*(\d+(?:\.\d+)?)', r'log10(\1)', expr)
        expr = re.sub(r'(\d+(?:\.\d+)?)\s*percent\s*of\s*(\d+(?:\.\d+)?)', r'(\1/100)*\2', expr)
        expr = re.sub(r'\s+', '', expr)

        allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed_names.update({'abs': abs, 'round': round, 'pow': pow, 'min': min, 'max': max})
        try:
            result = eval(expr, {"__builtins__": {}}, allowed_names)
            return f"{random.choice(intro)} {text.strip()} = {result}"
        except Exception as e:
            return f"Oops! I couldn't solve that. Please check your math expression. (Error: {e})"

    def _generate_code(self, text: str) -> str:
        """Generate basic Python function code from user request."""
        intro = [
            "Here's a Python function for you!",
            "Let me show you how to do that in Python:",
            "Here's some Python code that should help:",
            "Check out this Python function:",
            "Python to the rescue!"
        ]
        lower = text.lower()
        
        # Add support for 'add 2 numbers' or 'add two numbers'
        if re.search(r'(?:write\s+a\s+function\s+that\s+)?adds?\s+(?:2|two)\s+numbers?', lower):
            code = """def add_two_numbers(a, b):
    return a + b

# Example usage:
# result = add_two_numbers(5, 3)
# print(result)  # Output: 8"""
            return f"{random.choice(intro)}\n```python\n{code}\n```"
        
        # Add support for 'add N numbers' (N=3-10)
        match = re.search(r'add (\d+) numbers', lower)
        if match:
            n = int(match.group(1))
            if 3 <= n <= 10:
                args = ', '.join([f'a{i+1}' for i in range(n)])
                sum_expr = ' + '.join([f'a{i+1}' for i in range(n)])
                code = f'def add_{n}_numbers({args}):\n    return {sum_expr}'
                return f"{random.choice(intro)}\n```python\n{code}\n```"
        
        # Add support for 'count to N' or 'print numbers from X to Y'
        match = re.search(r'(count|print numbers) (to|from) (\d+)( to (\d+))?', lower)
        if match:
            start = 1
            end = 10
            if match.group(2) == 'from' and match.group(3):
                start = int(match.group(3))
                if match.group(5):
                    end = int(match.group(5))
                else:
                    end = start + 9
            elif match.group(2) == 'to' and match.group(3):
                end = int(match.group(3))
            code = f'for i in range({start}, {end+1}):\n    print(i)'
            return f"{random.choice(intro)}\n```python\n{code}\n```"
        
        # Simple patterns for common requests
        code_snippets = [
            (r'add two numbers',
             'def add_two_numbers(a, b):\n    return a + b'),
            (r'subtract two numbers',
             'def subtract_two_numbers(a, b):\n    return a - b'),
            (r'multiply two numbers',
             'def multiply_two_numbers(a, b):\n    return a * b'),
            (r'divide two numbers',
             'def divide_two_numbers(a, b):\n    if b == 0:\n        return "Cannot divide by zero"\n    return a / b'),
            (r'factorial',
             'def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)'),
            (r'reverse a string',
             'def reverse_string(s):\n    return s[::-1]'),
            (r'palindrome',
             'def is_palindrome(s):\n    return s == s[::-1]'),
            (r'fibonacci',
             'def fibonacci(n):\n    """Returns the nth Fibonacci number (0-indexed)\n    Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...\n    """\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    \n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n# Example usage:\n# print(fibonacci(0))  # 0\n# print(fibonacci(1))  # 1\n# print(fibonacci(2))  # 1\n# print(fibonacci(3))  # 2\n# print(fibonacci(4))  # 3\n# print(fibonacci(5))  # 5'),
            (r'fibonacci.*dynamic.*programming|dynamic.*programming.*fibonacci|optimized.*fibonacci',
             'def fibonacci_dp(n):\n    if n <= 1:\n        return n\n    \n    # Dynamic programming approach\n    dp = [0] * (n + 1)\n    dp[0], dp[1] = 0, 1\n    \n    for i in range(2, n + 1):\n        dp[i] = dp[i-1] + dp[i-2]\n    \n    return dp[n]\n\n# Time: O(n), Space: O(n)'),
            (r'longest.*common.*subsequence|lcs',
             'def longest_common_subsequence(text1, text2):\n    m, n = len(text1), len(text2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    \n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if text1[i-1] == text2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    \n    return dp[m][n]\n\n# Time: O(m*n), Space: O(m*n)'),
            (r'knapsack.*problem|knapsack',
             'def knapsack(values, weights, capacity):\n    n = len(values)\n    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n    \n    for i in range(1, n + 1):\n        for w in range(capacity + 1):\n            if weights[i-1] <= w:\n                dp[i][w] = max(dp[i-1][w], dp[i-1][w - weights[i-1]] + values[i-1])\n            else:\n                dp[i][w] = dp[i-1][w]\n    \n    return dp[n][capacity]\n\n# Time: O(n*capacity), Space: O(n*capacity)'),
            (r'edit.*distance|levenshtein',
             'def edit_distance(word1, word2):\n    m, n = len(word1), len(word2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    \n    for i in range(m + 1):\n        dp[i][0] = i\n    for j in range(n + 1):\n        dp[0][j] = j\n    \n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if word1[i-1] == word2[j-1]:\n                dp[i][j] = dp[i-1][j-1]\n            else:\n                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])\n    \n    return dp[m][n]\n\n# Time: O(m*n), Space: O(m*n)'),
            (r'coin.*change|minimum.*coins',
             'def coin_change(coins, amount):\n    dp = [float(\'inf\')] * (amount + 1)\n    dp[0] = 0\n    \n    for coin in coins:\n        for i in range(coin, amount + 1):\n            dp[i] = min(dp[i], dp[i - coin] + 1)\n    \n    return dp[amount] if dp[amount] != float(\'inf\') else -1\n\n# Time: O(amount * len(coins)), Space: O(amount)'),
            (r'sum of a list',
             'def sum_list(lst):\n    return sum(lst)'),
            (r'maximum in a list',
             'def max_in_list(lst):\n    return max(lst)'),
            (r'sort a list',
             'def sort_list(lst):\n    return sorted(lst)'),
            (r'prime',
             'def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True'),
        ]
        for pattern, code in code_snippets:
            if re.search(pattern, lower):
                return f"{random.choice(intro)}\n```python\n{code}\n```"
        
        # Fallback: generic function template
        return (f"{random.choice(intro)}\nHere's a generic Python function template you can adapt:\n"
                """```python
def my_function(args):
    # Your code here
    pass
```""")

    def _recall_fact(self, text: str) -> str:
        """Recall a fact about the user from memory."""
        # Use keywords from the question to search memory
        keywords = re.findall(r'(study|school|university|name|job|major|hobby|favorite|favourite|work|live)', text.lower())
        if not keywords:
            keywords = [w for w in text.lower().split() if len(w) > 2]
        facts = self.memory_bank.search(keywords[0] if keywords else '', limit=5)
        user_facts = [f for f in facts if hasattr(f, 'category') and f.category == 'user_fact']
        if user_facts:
            # Return the most recent fact
            fact = user_facts[0].content.get('text', '')
            return f"Here's what you told me before: {fact}"
        else:
            return "I don't recall you telling me that yet! If you'd like me to remember, just tell me again."

    def _weather_widget(self, text: str) -> dict:
        import re
        import requests
        WEATHERSTACK_API_KEY = "0bfd9b2ffc4866026f7381c4396ab17e"
        match = re.search(r'in ([A-Za-z ]+)', text)
        location = match.group(1).strip() if match else "New York"
        try:
            url = f"http://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={location}"
            resp = requests.get(url, timeout=6)
            data = resp.json()
            if "current" not in data:
                return {"response": f"Sorry, I couldn't get weather data for '{location}'. (Error: {data.get('error', {}).get('info', 'Unknown error')})"}
            temp = f"{data['current']['temperature']}°C"
            condition = data['current']['weather_descriptions'][0]
            icon_url = data['current']['weather_icons'][0] if data['current']['weather_icons'] else ""
            city = data['location']['name']
            return {
                "response": f"Here's the weather for {city}:",
                "widget": {
                    "type": "weather",
                    "location": city,
                    "temperature": temp,
                    "condition": condition,
                    "icon": icon_url
                }
            }
        except requests.Timeout:
            print("Weatherstack API request timed out.")
            return {
                "response": f"Sorry, the weather service timed out. Please try again later.",
            }
        except Exception as e:
            print(f"Weatherstack API exception: {e}")
            return {
                "response": f"Sorry, there was an error fetching the weather: {e}",
            }

class MemoryBank:
    "Manages agent's memory storage and retrieval"
    
    def __init__(self, db_path: str = ":memory:"):
        # Use check_same_thread=False for FastAPI thread safety
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
    
    def _init_db(self):
        "Initialize memory database"
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
        "Store a new memory"
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
        "Search memories by relevance"
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
        cursor = self.conn.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]

class TaskManager:
    "Manages agent tasks and execution"
    
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
        "Get all pending tasks sorted by priority"
        pending = [t for t in self.tasks.values() if not t.completed]
        return sorted(pending, key=lambda t: t.priority.value, reverse=True)
    
    def _update_queue(self):
        "Update task execution queue"
        pending = self.get_pending_tasks()
        self.task_queue = [t.id for t in pending if self._can_execute(t)]
    
    def _can_execute(self, task: Task) -> bool:
        "Check if task dependencies are satisfied"
        for dep_id in task.dependencies:
            if dep_id in self.tasks and not self.tasks[dep_id].completed:
                return False
        return True

class SkillRegistry:
    "Registry for agent skills"
    
    def __init__(self):
        self.skills: Dict[str, Callable] = {}
    
    def register(self, name: str, skill_func: Callable):
        "Register a new skill"
        self.skills[name] = skill_func
    
    def get_skill(self, name: str) -> Optional[Callable]:
        "Get skill by name"
        return self.skills.get(name)
    
    def list_skills(self) -> List[str]:
        "List all available skills"
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
    
    # Test 5: Math calculation
    response5 = agent.process_input("What is 2 + 2?")
    print(f"User: What is 2 + 2?")
    print(f"Agent: {response5}\n")

    response6 = agent.process_input("Calculate 5 * (3 + 2) - 10 / 2")
    print(f"User: Calculate 5 * (3 + 2) - 10 / 2")
    print(f"Agent: {response6}\n")

    response7 = agent.process_input("What is the square root of 16?")
    print(f"User: What is the square root of 16?")
    print(f"Agent: {response7}\n")

    response8 = agent.process_input("What is log base 10 of 100?")
    print(f"User: What is log base 10 of 100?")
    print(f"Agent: {response8}\n")

    response9 = agent.process_input("What is sin(90 degrees)?")
    print(f"User: What is sin(90 degrees)?")
    print(f"Agent: {response9}\n")

    response10 = agent.process_input("What is 2^3?")
    print(f"User: What is 2^3?")
    print(f"Agent: {response10}\n")

    response11 = agent.process_input("What is 100 / 0?")
    print(f"User: What is 100 / 0?")
    print(f"Agent: {response11}\n")

    response12 = agent.process_input("What is 100 / 0?")
    print(f"User: What is 100 / 0?")
    print(f"Agent: {response12}\n")
    
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