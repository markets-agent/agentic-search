from langchain_core.messages import HumanMessage
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.chains.text import get_summary_chain
from agentic_search.chains.web import get_concatenated_web_search_results_chain
from agentic_search.lib import log_if_debug
from agentic_search.graphs.web import get_search_the_web_react_graph


def get_scrape_and_summarize_webpages_from_single_query(query: str):
    """
    Get a chain that takes a query as an input and returns a concatenated string of summaries of the webpages.

    This works by:
    - scraping the first 3 web pages from the SERP based on the query
    - summarizing the web pages

    Returns a summary of a web search result.
    """
    web_search_res = get_concatenated_web_search_results_chain().invoke(
        {"query": query}
    )
    log_if_debug(f"Web search result: {web_search_res}")
    web_search_res["query"] = query
    return get_summary_chain("long-context").invoke(web_search_res)


def get_web_search_report(query: str):
    """
    Get a web search report for a given query.

    Returns a written Markdown report of the web search result.
    """
    invocation = get_search_the_web_react_graph().invoke(
        {"messages": [HumanMessage(content=query)]}
    )
    log_if_debug(f"Web search capability result: {invocation}")
    return invocation["messages"][-1].content
