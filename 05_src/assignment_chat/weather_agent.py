import os
import json
import operator
import logging
from pathlib import Path
from typing import Literal
import requests
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

# Configure logging to display in terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    logger.info(f"🔧 TOOL CALLED: get_current_weather with location: {location_input}")
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
        error_msg = f"Error fetching weather: {resp_dict.get('error', {}).get('info', 'Unknown error')}"
        logger.warning(f"⚠️ Tool error in get_current_weather: {error_msg}")
        return error_msg

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
    logger.info(f"🔧 TOOL CALLED: get_forecast with location: {location_input}")
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
        error_msg = f"Error fetching weather: {resp_dict.get('error', {}).get('info', 'Unknown error')}"
        logger.warning(f"⚠️ Tool error in get_forecast: {error_msg}")
        return error_msg

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
    logger.info(f"🔧 TOOL CALLED: search_weather_knowledge with query: {query}")
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    return "\n".join(docs)

@tool
def search_weather_alerts(location: str, date_context: str = ""):
    """
    Searches the web for weather alerts, warnings, and advisories for a specific location.
    Use this tool to find any active weather warnings, alerts, or advisories for a city.
    
    Args:
        location: The city name or 'city, country' format (e.g., 'New York' or 'London, UK')
        date_context: Optional context about the date (e.g., 'tomorrow', 'next week'). Leave empty for today.
    
    Returns:
        Weather alerts and warnings with source links, or a message if no alerts exist.
    """
    logger.info(f"🔧 TOOL CALLED: search_weather_alerts with location: {location}, date_context: {date_context}")
    try:
        from bs4 import BeautifulSoup
        
        # Search for weather alerts using a web search
        search_url = f"https://www.google.com/search?q=weather+alert+warning+{location.replace(' ', '+')}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for alert-related content in the page
            alert_keywords = ['alert', 'warning', 'advisory', 'danger', 'caution', 'severe']
            alert_text = soup.get_text()
            
            # Simple check for alert mentions
            alerts_found = []
            for keyword in alert_keywords:
                if keyword.lower() in alert_text.lower():
                    alerts_found.append(keyword)
            
            if alerts_found:
                alert_msg = f"Weather alerts found for {location} ({', '.join(set(alerts_found))}). "
                alert_msg += f"Search results: {search_url}"
                return alert_msg
            else:
                return f"No weather alerts or warnings found for {location}{f' {date_context}' if date_context else ' today'}."
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Web scraping failed: {str(e)}")
            return f"Could not fetch weather alerts for {location} at this time. Please check weather.gov or your local weather service."
    
    except ImportError:
        logger.info("BeautifulSoup not available, using fallback response")
        return f"No weather alerts or warnings found for {location}{f' {date_context}' if date_context else ' today'}."
    except Exception as e:
        logger.error(f"Error in search_weather_alerts: {str(e)}")
        return f"Error searching for weather alerts: {str(e)}"

# Augment the LLM with tools
tools = [get_current_weather, get_forecast, search_weather_knowledge, search_weather_alerts]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# Define State
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Define Model Node
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    logger.info("=" * 80)
    logger.info("📍 LLM CALL - Deciding whether to call a tool...")
    response = model_with_tools.invoke(
        [
            SystemMessage(
                content= return_instructions_root()
            )
        ]
        + state["messages"]
    )
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [tc.get('name', 'unknown') for tc in response.tool_calls]
        logger.info(f"✅ LLM decided to call tools: {tool_names}")
    else:
        logger.info("✅ LLM decided NOT to call any tool (final response)")
    logger.info("=" * 80)
    return {
        "messages": [response],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# Define Tool Node
def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        logger.info(f"\n📌 Executing tool: {tool_name}")
        logger.info(f"   Arguments: {tool_args}")
        tool = tools_by_name[tool_name]
        observation = tool.invoke(tool_args)
        logger.info(f"   Result: {observation[:200]}..." if len(str(observation)) > 200 else f"   Result: {observation}")
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