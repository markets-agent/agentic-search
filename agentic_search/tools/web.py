import os
import sys

from agentic_search.chains.text import (
    get_content_answers_to_query_chain,
    get_qa_summary_chain,
)
from agentic_search.functions.web import get_serp_links, get_webpages_soups_text_async

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.chains.web import (
    get_web_search_query_chain,
    get_web_search_queries_chain,
)
from agentic_search.lib import log_if_debug


async def get_agentic_quick_web_search_results_tool(
    query: str, estimated_number_of_searches: int
):
    """Make a quick web search prompted by a user query and output a nicely formatted and readable Markdown document.

    Args:
        query: The input user query
        estimated_number_of_searches: Estimated number of web searches to perform to answer the user query

    Returns:
        str: A formatted Markdown document containing the search results

    Use this tool if you need to quickly search the web for current information or information that is not in your knowledge base.
    """
    log_if_debug(f"invoking quickweb search tool with query: {query}")
    excluded_queries = []
    answer = ""
    for _ in range(estimated_number_of_searches):
        search_query = get_web_search_query_chain(excluded_queries).invoke(
            {"query": query}
        )
        links_to_scrape = []
        q_links = await get_serp_links(search_query["query"])
        if len(q_links) <= 0:
            continue
        links_to_scrape.extend(q_links)
        scraped_content = await get_webpages_soups_text_async(
            [x["href"] for x in links_to_scrape]
        )
        for item in scraped_content:
            content = answer + "\n\n" + item
            answer = get_qa_summary_chain().invoke(
                {
                    "content": content,
                    "query": query,
                }
            )
            answers_to_query = get_content_answers_to_query_chain().invoke(
                {"content": answer, "query": query}
            )
            if answers_to_query["fully_answered"] == "yes":
                return answer
            # re assigning `content` variable to a running summary chain on the whole content
            answer = get_qa_summary_chain().invoke(
                {
                    "content": answer,
                    "query": query,
                }
            )
        excluded_queries.append(search_query["query"])
    # this always returns some form of summary, regardless of it fully being answered
    return get_qa_summary_chain().invoke({"content": answer, "query": query})


async def get_agentic_thorough_web_search_results_tool(
    query: str, estimated_number_of_searches: int
):
    """Make a thorough web search prompted by a user query and output a nicely formatted and readable Markdown document.

    Args:
        query: The input user query
        estimated_number_of_searches: Estimated number of web searches to perform to answer the user query

    Returns:
        str: A formatted Markdown document containing the search results

    Use this tool if you need to thoroughly search the web for current information or information that is not in your knowledge base.
    """
    log_if_debug(f"invoking thorough web search tool with query: {query}")
    excluded_queries = []
    answer = ""
    for _ in range(estimated_number_of_searches):
        search_queries = get_web_search_queries_chain(excluded_queries).invoke(
            {"query": query}
        )
        for q in search_queries["queries"]:
            links_to_scrape = []
            links = await get_serp_links(q)
            if len(links) <= 0:
                continue
            links_to_scrape.extend(links)
            scraped_content = await get_webpages_soups_text_async(
                [x["href"] for x in links_to_scrape]
            )
            for item in scraped_content:
                content = answer + "\n\n" + item
                answer = get_qa_summary_chain().invoke(
                    {
                        "content": content,
                        "query": query,
                    }
                )
                answers_to_query = get_content_answers_to_query_chain().invoke(
                    {"content": answer, "query": query}
                )
                if answers_to_query["fully_answered"] == "yes":
                    return answer
                # re assigning `content` variable to a running summary chain on the whole content
                answer = get_qa_summary_chain().invoke(
                    {
                        "content": answer,
                        "query": query,
                    }
                )
                excluded_queries.append(q)
    # this always returns some form of summary, regardless of it fully being answered
    return get_qa_summary_chain().invoke({"content": answer, "query": query})


def get_web_search_tools():
    return [
        get_agentic_quick_web_search_results_tool,
        get_agentic_thorough_web_search_results_tool,
    ]
