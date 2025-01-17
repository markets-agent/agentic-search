import json
from langchain_core.messages import HumanMessage
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.lib import log_if_debug
from agentic_search.graphs.web import get_search_the_web_react_graph


async def get_web_search_results(query: str) -> dict:
    """
    Get a web search report for a given query using a LangGraph ReAct agent.

    Text search can be made in two ways:
    - quick search: a single search query is generated before iterative scraping
    - thorough search: multiple search queries are generated before iterative scraping
    In both cases, the results are returned as soon as the user's query is answered.

    Returns a written Markdown report of the web search result.
    """
    invocation = await get_search_the_web_react_graph().ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=query
                    + f"""Answer in JSON format: {{
                        "content": "your results as a string",
                        "metadata": "any additional metadata that was attached to the web search results, if any" | null,
                        "type": "text" | "video" (if the results are videos)
                        }}"""
                )
            ]
        }
    )
    log_if_debug(f"Web search capability result: {invocation}")
    return {
        "results": json.loads(
            invocation["messages"][-1]
            .content.replace("```json", "")
            .replace("```", "")
            .strip()
        )["content"],
        "metadata": json.loads(
            invocation["messages"][-1]
            .content.replace("```json", "")
            .replace("```", "")
            .strip()
        )["metadata"],
        "type": json.loads(
            invocation["messages"][-1]
            .content.replace("```json", "")
            .replace("```", "")
            .strip()
        )["type"],
    }
