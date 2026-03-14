import sys
from pathlib import Path

# Add parent directory to Python path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from weather_agent import agent
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr
from dotenv import load_dotenv
import os

from utils.logger import get_logger

_logs = get_logger(__name__)

llm = agent

load_dotenv('.secrets')

def weather_chat(message: str, history: list) -> str:
    langchain_messages = []
    n = 0
    _logs.debug(f"History: {history}")
    for msg_pair in history:
        if isinstance(msg_pair, (list, tuple)) and len(msg_pair) == 2:
            user_msg, assistant_msg = msg_pair
            if user_msg:
                langchain_messages.append(HumanMessage(content=user_msg))
            if assistant_msg:
                langchain_messages.append(AIMessage(content=assistant_msg))
                n += 1
    langchain_messages.append(HumanMessage(content=message))

    state = {
        "messages": langchain_messages,
        "llm_calls": n
    }

    response = llm.invoke(state)
    return response['messages'][len(response['messages']) - 1].content

# --- Gradio Chat Interface ---
demo = gr.ChatInterface(
    fn=weather_chat,
    title="Weather Agent Chatbot",
    description="Ask me about the current weather or forecast for any city in the world!",
    examples=[
        "What's the weather like in Tokyo right now?",
        "Give me a forecast of tomorrow for London, UK.",
        "Should I bring an umbrella in Paris today?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
