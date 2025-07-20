from backend.aiAgent import AIAgent, Priority, TaskManager, SkillRegistry
import json

# Create your first agent
agent = AIAgent(
    name="Assistant",
    personality={
        "traits": ["helpful", "friendly", "analytical"],
        "learning_rate": 0.2,
        "curiosity": 0.8
    }
)

print("Agent created successfully!")
print(f"Agent name: {agent.name}")
print(f"Current state: {agent.state.value}")

## 2. Basic Interactions

# Simple conversation
print("\n--- Basic Conversation ---")
response = agent.process_input("Hello! How are you today?")
print(f"You: Hello! How are you today?")
print(f"Agent: {response}")

# Ask a question
response = agent.process_input("What can you help me with?")
print(f"\nYou: What can you help me with?")
print(f"Agent: {response}")

# Give a command
response = agent.process_input("Analyze the sentiment of: I love programming!")
print(f"\nYou: Analyze the sentiment of: I love programming!")
print(f"Agent: {response}")

## 3. Working with Goals and Tasks

print("\n--- Goals and Tasks ---")

# Add goals for the agent
agent.add_goal("Help user with daily tasks", Priority.HIGH)
agent.add_goal("Learn user preferences", Priority.MEDIUM)
agent.add_goal("Improve response accuracy", Priority.LOW)

# Check pending tasks
pending_tasks = agent.task_manager.get_pending_tasks()
print(f"Number of pending tasks: {len(pending_tasks)}")

# View goals
print(f"Number of goals: {len(agent.goals)}")
for goal in agent.goals:
    print(f"  - {goal.description} (Priority: {goal.priority.name})")

## 4. Using Built-in Skills

print("\n--- Built-in Skills Demo ---")

# Text Analysis
text_to_analyze = """
Artificial Intelligence is transforming the world. Machine learning models 
are becoming more sophisticated every day. The future looks bright for AI technology!
"""
response = agent.process_input(f"Analyze this text: {text_to_analyze}")
print(f"Analysis result:\n{response}")

# Entity Extraction
response = agent.process_input(
    "Extract entities from: Sarah Johnson met with CEO Mark Chen in New York on 03/15/2024 to discuss the $5M investment deal."
)
print(f"\nEntity extraction:\n{response}")

## 5. Memory and Learning

print("\n--- Memory and Learning ---")

# Teach the agent something
agent.process_input("Remember that my favorite programming language is Python")
agent.process_input("I prefer detailed explanations over brief summaries")
agent.process_input("My name is Alex")

# Test memory recall
response = agent.process_input("What do you remember about me?")
print(f"\nMemory test: {response}")

# Check agent's experience
status = agent.get_status()
print(f"\nAgent Experience: {status['experience']} points")
print(f"Skill Levels: {status['skills']}")

## 6. Adding Custom Skills

print("\n--- Custom Skills ---")

# Define a custom skill for weather (mock example)
def weather_skill(text: str) -> str:
    """Custom skill to handle weather queries"""
    if "weather" in text.lower():
        # In real implementation, you'd call a weather API here
        return "Based on current patterns, it appears to be partly cloudy with a temperature around 72°F. Would you like more detailed forecast information?"
    return "I can help you check the weather. Just ask about the weather in any location!"

# Define a custom skill for calculations
def calculator_skill(text: str) -> str:
    """Custom skill for basic calculations"""
    import re
    
    # Look for math expressions
    math_pattern = r'(\d+)\s*([\+\-\*\/])\s*(\d+)'
    match = re.search(math_pattern, text)
    
    if match:
        num1, operator, num2 = match.groups()
        num1, num2 = float(num1), float(num2)
        
        operations = {
            '+': num1 + num2,
            '-': num1 - num2,
            '*': num1 * num2,
            '/': num1 / num2 if num2 != 0 else "Error: Division by zero"
        }
        
        result = operations.get(operator, "Unknown operation")
        return f"The result of {num1} {operator} {num2} = {result}"
    
    return "I can help with basic calculations. Try asking me to calculate something like '5 + 3' or '10 * 4'."

# Register custom skills
agent.skill_registry.register("weather", weather_skill)
agent.skill_registry.register("calculator", calculator_skill)

# List all available skills
print("Available skills:", agent.skill_registry.list_skills())

# Test custom skills
response = agent.skill_registry.get_skill("weather")("What's the weather like today?")
print(f"\nWeather skill: {response}")

response = agent.skill_registry.get_skill("calculator")("Calculate 25 * 4")
print(f"\nCalculator skill: {response}")

## 7. Creating a Chat Interface

print("\n--- Interactive Chat Interface ---")

def chat_with_agent(agent):
    """Simple chat interface for interacting with the agent"""
    print("\n🤖 AI Agent Chat Interface")
    print("Type 'quit' to exit, 'status' to see agent info, 'help' for commands")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Agent: Goodbye! It was nice chatting with you.")
            break
        elif user_input.lower() == 'status':
            status = agent.get_status()
            print(f"\nAgent Status:")
            print(f"  Name: {status['name']}")
            print(f"  State: {status['state']}")
            print(f"  Experience: {status['experience']}")
            print(f"  Active Tasks: {status['active_tasks']}")
            print(f"  Memories: {status['memories']}")
            continue
        elif user_input.lower() == 'help':
            print("\nAvailable commands:")
            print("  - Regular text: Chat with the agent")
            print("  - 'status': View agent status")
            print("  - 'quit': Exit the chat")
            print("  - Ask questions starting with 'what', 'how', etc.")
            print("  - Give commands like 'analyze', 'summarize', etc.")
            continue
        
        # Process regular input
        response = agent.process_input(user_input)
        print(f"Agent: {response}")

# Uncomment to run the interactive chat
# chat_with_agent(agent)

## 8. Advanced Usage Examples

print("\n--- Advanced Examples ---")

# Example 1: Building a Personal Assistant
class PersonalAssistant(AIAgent):
    """Extended AI Agent for personal assistant tasks"""
    
    def __init__(self, name: str, user_name: str):
        super().__init__(name)
        self.user_name = user_name
        self._register_assistant_skills()
    
    def _register_assistant_skills(self):
        """Register personal assistant specific skills"""
        self.skill_registry.register("reminder", self._set_reminder)
        self.skill_registry.register("schedule", self._check_schedule)
    
    def _set_reminder(self, text: str) -> str:
        # In practice, integrate with calendar API
        return f"I've set a reminder for you, {self.user_name}. I'll make sure to notify you!"
    
    def _check_schedule(self, text: str) -> str:
        # In practice, integrate with calendar API
        return f"Let me check your schedule, {self.user_name}. You appear to be free this afternoon."

# Create personal assistant
assistant = PersonalAssistant("Jarvis", "Tony")
response = assistant.process_input("Set a reminder for my meeting tomorrow")
print(f"Personal Assistant: {response}")

# Example 2: Saving and Loading Agent State
def save_agent_state(agent, filename="agent_state.json"):
    """Save agent state to file"""
    state = {
        "name": agent.name,
        "experience": agent.experience_points,
        "skills": dict(agent.skill_levels),
        "personality": agent.personality,
        "conversation_count": len(agent.conversation_history)
    }
    
    with open(filename, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"Agent state saved to {filename}")

def load_agent_state(filename="agent_state.json"):
    """Load agent state from file"""
    try:
        with open(filename, 'r') as f:
            state = json.load(f)
        
        # Create new agent with loaded state
        agent = AIAgent(
            name=state["name"],
            personality=state["personality"]
        )
        agent.experience_points = state["experience"]
        agent.skill_levels.update(state["skills"])
        
        print(f"Agent state loaded from {filename}")
        return agent
    except FileNotFoundError:
        print("No saved state found. Creating new agent.")
        return None

# Save current agent state
save_agent_state(agent)

# Example 3: Multi-Agent Collaboration
def create_specialist_team():
    """Create a team of specialized agents"""
    
    # Create specialized agents
    analyst = AIAgent("DataAnalyst", {
        "traits": ["analytical", "precise", "logical"],
        "learning_rate": 0.3,
        "curiosity": 0.6
    })
    
    creative = AIAgent("CreativeWriter", {
        "traits": ["creative", "imaginative", "expressive"],
        "learning_rate": 0.2,
        "curiosity": 0.9
    })
    
    coordinator = AIAgent("ProjectCoordinator", {
        "traits": ["organized", "efficient", "collaborative"],
        "learning_rate": 0.25,
        "curiosity": 0.7
    })
    
    return {
        "analyst": analyst,
        "creative": creative,
        "coordinator": coordinator
    }

# Create a team
team = create_specialist_team()
print("\nSpecialist team created:")
for role, agent in team.items():
    print(f"  - {role}: {agent.name}")

# Final status check
print("\n--- Final Agent Status ---")
final_status = agent.get_status()
print(json.dumps(final_status, indent=2))

print("\n Demo complete! You now know how to use the AI Agent framework.")
