import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from weather_agent import agent

# --- Gradio Chat Interface ---

def chat(user_message: str, history: list):
    """
    Takes the new user message and full chat history,
    builds the LangGraph message list, invokes the agent,
    and returns the assistant's reply.
    """
    # Convert Gradio history to LangChain messages
    lc_messages = []
    for human, assistant in history:
        lc_messages.append(HumanMessage(content=human))
        lc_messages.append(AIMessage(role="assistant", content=assistant))

    lc_messages.append(HumanMessage(content=user_message))

    result = agent.invoke({"messages": lc_messages})

    # The last message is the assistant's final reply
    reply = result["messages"][-1].content
    return reply


demo = gr.ChatInterface(
    fn=chat,
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
