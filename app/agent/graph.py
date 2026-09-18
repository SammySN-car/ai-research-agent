"""LangGraph agent — stateful, multi-step research workflow."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from app.agent.tools import TOOLS
from app.config import (
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TEMPERATURE,
)
from app.rag.retrieval import retrieve_and_format


# ── State Definition ──────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    tools_used: list[str]


# ── Node Functions ────────────────────────────────────
def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant document chunks for the user's question."""
    last_message = state["messages"][-1]
    question = last_message.content
    context, _sources = retrieve_and_format(question)
    return {"context": context}


def agent_node(state: AgentState) -> dict:
    """Call the LLM with tools bound, passing retrieved context."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=(
            SecretStr(OPENROUTER_API_KEY)
            if OPENROUTER_API_KEY is not None
            else None
        ),
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Research Agent",
        },
    )

    llm_with_tools =llm.bind_tools(TOOLS)

    context = state.get("context", "No context available.")
    system_msg = SystemMessage(content=(
        "You are a research assistant. Answer questions using the provided "
        "context and tools. Always cite sources when using document context.\n\n"
        f"Retrieved Context:\n{context}"
    ))

    messages = [system_msg] + list(state["messages"])
    response = llm_with_tools.invoke(messages)

    # Track tool usage
    tools_used = list(state.get("tools_used", []))
    if response.tool_calls:
        for tc in response.tool_calls:
            tools_used.append(tc["name"])

    return {"messages": [response], "tools_used": tools_used}


def tool_node(state: AgentState) -> dict:
    """Execute any tool calls from the last LLM response."""
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        # Find the matching tool by name
        tool_fn = next(t for t in TOOLS if t.name == tool_call["name"])
        result = tool_fn.invoke(tool_call["args"])
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_results}


# ── Routing Function ─────────────────────────────────
def should_continue(state: AgentState) -> Literal["use_tool", "__end__"]:
    """Decide whether to execute tools or finish."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "__end__"


# ── Graph Assembly ────────────────────────────────────
def create_agent(checkpointer: MemorySaver | None = None) -> CompiledStateGraph:
    """Build and compile the LangGraph agent with optional checkpoint memory."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("use_tool", tool_node)

    # Define flow
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("use_tool", "agent")

    return graph.compile(checkpointer=checkpointer)


# ── Singleton ─────────────────────────────────────────
_memory = MemorySaver()
_agent: CompiledStateGraph | None = None


def get_agent() -> CompiledStateGraph:
    """Return a cached compiled agent with active thread checkpointer."""
    global _agent
    if _agent is None:
        _agent = create_agent(checkpointer=_memory)
    return _agent
