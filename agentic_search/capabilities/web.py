from langchain_core.messages import HumanMessage
import os
import sys
import json
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.lib import log_if_debug
from agentic_search.graphs.web import get_search_the_web_react_graph


async def get_web_search_results(query: str):
    """
    Get a web search report for a given query using a LangGraph ReAct agent.

    Search can be made in two ways:
    - quick search: a single search query is generated before iterative scraping
    - thorough search: multiple search queries are generated before iterative scraping
    In both cases, the results are returned as soon as the user's query is answered.

    Returns a written Markdown report of the web search result.
    """
    invocation = await get_search_the_web_react_graph().ainvoke(
        {"messages": [
            HumanMessage(content=query + f"""Answer in JSON format: {{"content": "your results"}}""")
        ]}
    )
    log_if_debug(f"Web search capability result: {invocation}")
    return {
        "results": json.loads(invocation["messages"][-1].content)["content"]
    }
