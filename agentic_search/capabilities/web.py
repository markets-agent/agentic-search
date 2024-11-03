from langchain_core.messages import HumanMessage
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.lib import log_if_debug
from agentic_search.graphs.web import get_search_the_web_react_graph


def get_web_search_capability(query: str):
    """
    Get a web search summary result for a given query.

    Returns a string summary of the web search result.
    """
    invocation = get_search_the_web_react_graph().invoke(
        {"messages": [HumanMessage(content=query)]}
    )
    log_if_debug(f"Web search capability result: {invocation}")
    return invocation["messages"][-1].content
