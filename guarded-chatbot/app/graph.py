"""LangGraph workflow: START -> chat (guarded LLM call) -> END, with memory."""
from langchain_core.messages import AIMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.config import MAX_HISTORY_MESSAGES, OPENAI_MODEL
from app.guardrails import GuardedLLM


def build_graph():
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)
    guarded = GuardedLLM(llm)

    def chat_node(state: MessagesState):
        history = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,  # count messages, not tokens
            max_tokens=MAX_HISTORY_MESSAGES,
            start_on="human",
            include_system=False,
        )
        reply = guarded.generate(history)
        return {"messages": [AIMessage(content=reply)]}

    builder = StateGraph(MessagesState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=MemorySaver())