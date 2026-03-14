This application lets users ask natural language questions about current weather conditions, forecasts, and weather alerts for any city in the world. The agent automatically selects the appropriate tool based on the user's query.

## Project Structure

The project consists of three files:

- **weather_agent.py** - Defines the tools, language model, and LangGraph agent workflow.
- **app.py** - Provides the Gradio chat interface and is the entry point for the application.
- **prompts.py** - Contains the system prompt and instructions for the agent.

## Available Tools

The agent has four tools to answer weather questions:

1. **get_current_weather** - Retrieves current weather conditions (temperature, humidity, wind speed, etc.)
2. **get_forecast** - Fetches a 7-day weather forecast for a location
3. **search_weather_knowledge** - Searches a knowledge base for weather tips and safety advice
4. **search_weather_alerts** - Searches for active weather warnings and alerts for a location

## Key Features

- Intelligent tool selection based on user queries
- Real-time logging of all tool calls and LLM decisions to the terminal
- Vector database (ChromaDB) for semantic search of weather knowledge
- Web scraping integration for weather alert detection
- Gradio chat interface for easy interaction

## Setup

Create a `.secrets` file with:
```
API_GATEWAY_KEY=<your_api_gateway_key>
WEATHERSTACK_API_KEY=<your_weatherstack_api_key>
```

## Running the Application

```bash
python app.py
```

The application will launch at http://localhost:7860
