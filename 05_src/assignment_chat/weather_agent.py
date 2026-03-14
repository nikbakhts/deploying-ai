import os
import json
import operator
from pathlib import Path
from typing import Literal
import requests
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

# Disable LangSmith tracing to avoid authentication errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from prompts import return_instructions_root
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Load environment variables from .secrets file
secrets_path = Path(__file__).parent.parent / '.secrets'
load_dotenv(secrets_path)

# Initialize the chat model
model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0.7,
    base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1', 
    api_key='any value',
    default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')}
)

# setup chromadb vector store
embedding_fn = OpenAIEmbeddingFunction(
        api_key = "any value",
        model_name="text-embedding-3-small",
        api_base='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1',
        default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')})


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("weather_knowledge", embedding_function=embedding_fn)

# Define tools
base_url = "http://api.weatherstack.com"
@tool
def get_current_weather(location_input:str):
    """
    Returns current weather from the WeatherStack API.
    For the location_input parameter, use the city name (e.g. 'London') or, 
    if available, the 'city, country' format (e.g. 'London, UK') 
    for more accurate results.
    """
    url = f"{base_url}/current"
    params = {
        "access_key": os.getenv('WEATHERSTACK_API_KEY'),
        "type": "City",
        "query": location_input,
        "units": "m"
    }
    response = requests.get(url, params=params)
    resp_dict = json.loads(response.text)

    if not resp_dict.get("success", True):
        return f"Error fetching weather: {resp_dict.get('error', {}).get('info', 'Unknown error')}"

    location = resp_dict.get("location", {})
    current = resp_dict.get("current", {})

    report = (
        f"Location: {location.get('name')}, {location.get('country')}\n"
        f"Temperature: {current.get('temperature')}°C\n"
        f"Description: {', '.join(current.get('weather_descriptions', []))}\n"
        f"Humidity: {current.get('humidity')}%\n"
        f"Wind Speed: {current.get('wind_speed')} km/h\n"
        f"Feels Like: {current.get('feelslike')}°C\n"
    )
    return report

@tool
def get_forecast(location_input:str):
    """
    Returns n forecast from the WeatherStack API.
    For the location_input parameter, use the city name (e.g. 'London') or, 
    if available, the 'city, country' format (e.g. 'London, UK') 
    for more accurate results.    
    """
    url = f"{base_url}/forecast"
    params = {
        "access_key": os.getenv('WEATHERSTACK_API_KEY'),
        "type": "City",
        "query": location_input,
        "forecast_days": 7,
        "hourly": 1,
        "interval": 3,
        "units": "m"
    }
    response = requests.get(url, params=params)
    resp_dict = json.loads(response.text)

    if not resp_dict.get("success", True):
        return f"Error fetching weather: {resp_dict.get('error', {}).get('info', 'Unknown error')}"

    location = resp_dict.get("location", {})
    current = resp_dict.get("current", {})

    report = (
        f"Location: {location.get('name')}, {location.get('country')}\n"
        f"Temperature: {current.get('temperature')}°C\n"
        f"Description: {', '.join(current.get('weather_descriptions', []))}\n"
        f"Humidity: {current.get('humidity')}%\n"
        f"Wind Speed: {current.get('wind_speed')} km/h\n"
        f"Feels Like: {current.get('feelslike')}°C\n"
    )
    return report

@tool
def search_weather_knowledge(query: str):
    """
    Searches a knowledge base of weather tips, safety advice, and 
    general weather-related guidance using semantic similarity.
    Use this when the user asks for advice, tips, or explanations 
    rather than current conditions or forecasts.
    """
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    return "\n".join(docs)

# Augment the LLM with tools
tools = [get_current_weather, get_forecast,search_weather_knowledge]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# Define State
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Define Model Node
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content= return_instructions_root()
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# Define Tool Node
def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# Define End Logic
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Build workflow
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")
agent = agent_builder.compile()