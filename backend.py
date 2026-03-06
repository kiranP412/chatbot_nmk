# backend_gemini.py

from typing import TypedDict, Annotated, List
import os

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import BaseMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a fast, helpful AI assistant. Answer in short, clear responses unless the user explicitly asks for detail.",
)

# LangGraph State Definition


class MessageState(TypedDict):
    """
    Conversation state for LangGraph.

    messages:
      A list of LangChain BaseMessage objects (HumanMessage, AIMessage, SystemMessage, etc.).
      We use add_messages so nodes can return {"messages": [new_message]} and LangGraph
      automatically appends them.
    """
    messages: Annotated[List[BaseMessage], add_messages]

# LLM Setup (Gemini 2.5 Flash)

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0.4,
    max_output_tokens=512,   # keep answers short -> faster
)


def chat_node(state: MessageState) -> MessageState:
    """
    Core chat node. Takes the conversation messages and returns the LLM response.
    """
    messages = state["messages"]

    # Ensure we always have a system prompt at the beginning
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=DEFAULT_SYSTEM_PROMPT)] + list(messages)

    response = llm.invoke(messages)
    return {"messages": [response]}


checkpointer = MemorySaver()

# Build LangGraph Graph

graph = StateGraph(MessageState)

graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# This is what the Streamlit app imports
workflow = graph.compile(checkpointer=checkpointer)

