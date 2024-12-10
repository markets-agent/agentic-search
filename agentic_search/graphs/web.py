from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.nodes.web import get_web_search_agent_node, get_web_search_tools


def get_search_the_web_react_graph():
    """Get the react graph for searching the web."""
    builder = StateGraph(MessagesState)

    # add nodes to the graph
    builder.add_node(
        "web_search_agent",
        get_web_search_agent_node,
    )
    builder.add_node(
        "tools",
        ToolNode(get_web_search_tools()),
    )

    # add edges to the graph
    builder.add_edge(START, "web_search_agent")
    # here is the branching logic
    builder.add_conditional_edges(
        "web_search_agent",
        # if the last message from the agent is a tool use, continue to the tool node,
        # otherwise continue to the end of the graph (in this case, the agent itself)
        tools_condition,
    )
    # this simple edge, with the conditional edge, is actually what enables the ReAct pattern:
    # this change creates a powerful agentic loop!
    builder.add_edge("tools", "web_search_agent")

    return builder.compile()
