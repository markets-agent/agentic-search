import json
import os
import sys
import time

from agentic_search.chains.text import (
    get_content_answers_to_query_chain,
    get_qa_summary_chain,
)
from agentic_search.functions.web import (
    get_news,
    get_serp_links,
    get_videos,
    get_webpages_soups_text_async,
    get_webpages_text_using_scraping_service,
)

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.chains.web import (
    get_route_search_type_chain,
    get_web_search_query_chain,
    get_web_search_queries_chain,
)
from agentic_search.lib import get_websearch_llm_provider, log_if_debug


async def get_web_search_results_tool(
    query: str, estimated_number_of_searches: int, is_thorough: bool
):
    """Make a quick web search prompted by a user query and output a nicely formatted and readable Markdown document.

    Args:
        query: The input user query
        estimated_number_of_searches: Estimated number of web searches to perform to answer the user query
        is_thorough: Whether to perform each web search in thorough or quick mode (search with several search engine queries at a time VS a quick search with one search engine query at a time)

    Returns:
        str: A formatted Markdown document containing the search results

    Use this tool if you need to quickly search the web for current information or information that is not in your knowledge base.
    """
    log_if_debug(
        f"invoking web search tool with query: {query}, and {estimated_number_of_searches}, thorough: {is_thorough}"
    )

    search_type = get_route_search_type_chain().invoke({"query": query})["search_type"]

    if search_type == "news":
        news = await get_news(query)
        # concatenate `date` `title` `body` prop of each news item
        news_content = "\n\n---\n\n".join([f"{n['date']}\n\n{n['title']}\n\n{n['body']}" for n in news])
        return {
            "content": f"{news_content}",
            "metadata": "",
            "type": "text",
        }

    if search_type == "video":
        vids = await get_videos(query)
        videos_with_most_views = sorted(
            vids, key=lambda x: x["statistics"]["viewCount"], reverse=True
        )
        return {
            "content": f"{videos_with_most_views[0]['content']}",
            "metadata": f"{videos_with_most_views[0]['title']}",
            "type": "video",
        }

    excluded_queries = []
    answer = ""
    start_time = time.time()
    for _ in range(estimated_number_of_searches):
        # check if we have an answer and if we've exceeded 30 seconds
        if answer.strip() != "" and (time.time() - start_time) > 30:
            return answer

        search_queries = (
            get_web_search_queries_chain(excluded_queries).invoke({"query": query})
            if is_thorough
            else {
                "queries": [
                    get_web_search_query_chain(excluded_queries).invoke(
                        {"query": query}
                    )["query"]
                ]
            }
        )
        for q in search_queries["queries"]:
            links_to_scrape = []
            links = await get_serp_links(q)
            if len(links) <= 0:
                continue
            # concatenate `body` prop of each SERP link
            answer = "\n\n".join([x["body"] for x in links])
            links_to_scrape.extend(links)
            scraped_content = []
            if int(os.getenv("USE_EXTERNAL_SCRAPING_SERVICE")) == 1:
                log_if_debug(f"using external scraping service")
                scraped_content = await get_webpages_text_using_scraping_service(
                    [x["href"] for x in links_to_scrape]
                )
            else:
                log_if_debug(f"using local scraping service")
                scraped_content = await get_webpages_soups_text_async(
                    [x["href"] for x in links_to_scrape]
                )
            # prepend initial content to scraped content array
            for item in scraped_content:
                content = answer + "\n\n" + item
                log_if_debug(f"content input to 1st qa summary chain: {content}")
                tmp_answer = json.loads(get_qa_summary_chain(
                    "default", get_websearch_llm_provider()
                ).invoke(
                    {
                        "content": content,
                        "query": query,
                    }
                ))["content"]
                answers_to_query = get_content_answers_to_query_chain(
                    get_websearch_llm_provider()
                ).invoke({"content": tmp_answer, "query": query})
                log_if_debug(f"tmp_answer: {tmp_answer}")
                log_if_debug(f"answers_to_query: {answers_to_query}")
                if answers_to_query["answered"] == "yes":
                    return {
                        "content": tmp_answer,
                        "metadata": "",
                        "type": "text",
                    }
                # running summary chain on the whole content
                answer = json.loads(get_qa_summary_chain(
                    "default", get_websearch_llm_provider()
                ).invoke(
                    {
                        "content": tmp_answer,
                        "query": query,
                    }
                ))["content"]
                log_if_debug(f"web search answer: {answer}")
                excluded_queries.append(q)
    # this always returns some form of summary, regardless of it fully being answered
    return {
        "content": json.loads(get_qa_summary_chain("default", get_websearch_llm_provider()).invoke(
            {"content": answer, "query": query}
        ))["content"],
        "metadata": "",
        "type": "text",
    }


def get_web_search_tools():
    return [
        get_web_search_results_tool,
    ]
