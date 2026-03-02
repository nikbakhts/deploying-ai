This application lets users ask natural language questions about current weather conditions and forecasts for any city in the world.

The project is split into three files:

- **`weather_agent.py`** — defines the tools, language model, and LangGraph agent. Exposes a compiled `agent` object.
- **`app.py`** — imports `agent` from `weather_agent.py` and owns the Gradio chat interface. This is the entry point.
- **`prompts`** - defines the system prompt, the instruction, in a separate file.
