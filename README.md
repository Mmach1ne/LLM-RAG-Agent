# AI Agent Framework

A modular, extensible Python framework for building advanced AI agents with memory, skills, goals, and interactive capabilities.

---

## Features

- **Agent Personality:** Define traits, learning rate, and curiosity.
- **Task Management:** Add, prioritize, and track goals and tasks.
- **Memory System:** Store and recall interactions and learned patterns using SQLite.
- **Skill Registry:** Register built-in and custom skills (e.g., text analysis, entity extraction, calculator, weather).
- **Interactive Chat:** Command-line chat interface for live interaction.
- **Extensible:** Easily add new skills, integrate APIs, or create specialized agent subclasses.
- **Persistence:** Save and load agent state.
- **Multi-Agent Collaboration:** Create teams of specialist agents.

---

## Getting Started

### 1. Clone or Download

Place `aiAgent.py` and `implementation.py` in the same directory.

### 2. Install Requirements

No external dependencies are required for the core framework.  
For advanced integrations (OpenAI, web scraping), install:
```bash
pip install openai requests beautifulsoup4
```

### 3. Run the Demo

```bash
python implementation.py
```

---

## Usage

### Create an Agent

```python
from aiAgent import AIAgent, Priority

agent = AIAgent(
    name="Assistant",
    personality={
        "traits": ["helpful", "analytical"],
        "learning_rate": 0.2,
        "curiosity": 0.8
    }
)
```

### Interact with the Agent

```python
response = agent.process_input("Hello! How are you today?")
print(response)
```

### Add Goals and Tasks

```python
agent.add_goal("Help user with daily tasks", Priority.HIGH)
```

### Register Custom Skills

```python
def weather_skill(text):
    return "It's sunny!"

agent.skill_registry.register("weather", weather_skill)
```

### Use the Chat Interface

Uncomment the line in `implementation.py`:
```python
# chat_with_agent(agent)
```
Then run the script for an interactive session.

---

## Advanced Examples

- **Personal Assistant:** Extend `AIAgent` to create specialized assistants.
- **Save/Load State:** Use provided functions to persist agent progress.
- **Multi-Agent Teams:** Instantiate and coordinate multiple agents.

---

## Best Practices

- Save agent state regularly.
- Clean up old memories as needed.
- Group related skills into modules.
- Handle errors gracefully.
- Respect privacy and sensitive data.

---

## Integration Ideas

- **OpenAI GPT:** Add advanced language skills.
- **Database Access:** Query and summarize data.
- **Web Scraping:** Fetch and process online information.

See the "Integration Ideas" section in `implementation.py` for code templates.

---

## License

MIT License (add your own license if needed)

---

## Author

Your Name Here

---

## Acknowledgments

- Inspired by modular AI agent architectures.
- Uses Python standard library for maximum compatibility.

---

Enjoy building with your AI Agent