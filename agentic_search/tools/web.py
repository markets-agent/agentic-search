import asyncio
import os
import random
import sys

from agentic_search.chains.text import get_summary_chain
from agentic_search.functions.web import get_serp_links, get_webpages_soups_text

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.chains.web import get_web_search_queries_chain
from agentic_search.lib import log_if_debug


async def get_agentic_web_search_results_tool(query: str):
    """Search the web for the given query and output a nicely formatted and readable Markdown document.

    1st argument: `query`

    Use this tool if you need to search the web for current information or information that is not in your knowledge base.
    """
    log_if_debug(f"invoking web search tool with query: {query}")
    search_queries = get_web_search_queries_chain().invoke({"query": query})
    content = ''
    links_to_scrape = []
    for query in search_queries["queries"]:
        q_links = await get_serp_links(query)
        for link in q_links:
            content += link["body"]
            content += "\n\n---\n\n"
        if len(q_links) > 0:
            links_to_scrape.extend(q_links)
        # wait a random amount of time between 1-2 seconds before the next query
        await asyncio.sleep(1 + random.random())
    scraped_content = get_webpages_soups_text(
        [x["href"] for x in links_to_scrape], query
    )
    content += scraped_content
    return get_summary_chain("long-context").invoke({"content": content, "query": query})


def get_web_search_tools():
    return [get_agentic_web_search_results_tool]
