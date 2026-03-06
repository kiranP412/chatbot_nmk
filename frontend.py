import uuid
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from backend import workflow



# Helpers


def get_or_create_thread_id() -> str:
    """
    Generate a unique thread_id per chat session (browser tab).
    Used by LangGraph's MemorySaver to keep separate conversations.
    """
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())
    return st.session_state["thread_id"]


def init_message_history():
    """
    Initialize in-memory chat history used by Streamlit and passed into LangGraph.
    """
    if "message_history" not in st.session_state:
        st.session_state["message_history"]: List[BaseMessage] = []


def render_chat_history():
    """
    Render all previous messages in the chat UI.
    """
    for msg in st.session_state["message_history"]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)


def stream_text(text: str):
    """
    Simple generator to stream LLM output token-by-token into Streamlit.
    """
    for token in text.split():
        yield token + " "

# Streamlit Page Setup
load_dotenv()

st.set_page_config(
    page_title=" NMK Chatbot",

    layout="centered",
)

st.title(" NMK CHAT BOT")
st.caption("Runs fully ")


# Sidebar
with st.sidebar:
    st.subheader("Session")
    thread_id = get_or_create_thread_id()
    st.code(f"thread_id: {thread_id}", language="bash")

    if st.button("🧹 Start New Conversation"):
        st.session_state["message_history"] = []
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.markdown(
        "💡 Each browser tab has its own thread_id. "
        "Conversations are kept in memory while the app is running."
    )


# Initialize state
init_message_history()

# Show previous messages
render_chat_history()


# Chat Input


user_input = st.chat_input("Type your question here...")

if user_input:
    # 1. Show user message in UI
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Append to in-memory history
    st.session_state["message_history"].append(HumanMessage(content=user_input))

    # 3. Build config for LangGraph
    thread_id = get_or_create_thread_id()

    CONFIG = {
        "configurable": {
            "thread_id": thread_id,
        },
        "run_name": f"chat_llm_{thread_id}",
        "metadata": {
            "thread_id": thread_id,
        },
    }

    # 4. Call workflow
    with st.chat_message("assistant"):
        result_state = workflow.invoke(
            {"messages": st.session_state["message_history"]},
            config=CONFIG,
        )

        messages = result_state["messages"]
        last_message = messages[-1] if messages else None

        if isinstance(last_message, AIMessage):
            answer_text = (
                last_message.content
                if isinstance(last_message.content, str)
                else str(last_message.content)
            )
        else:
            answer_text = "I couldn't generate a response. Please try again."

        # Stream the AI response
        _ = st.write_stream(stream_text(answer_text))

    # 5. Save AI message to history
    st.session_state["message_history"].append(AIMessage(content=answer_text))