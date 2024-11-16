from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
import os
import sys
from yollama import get_llm

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.functions.web import get_serp_links, scrape_webpage_text
from agentic_search.prompts.text import (
    get_summary_prompt,
)
from agentic_search.prompts.web import (
    get_user_query_expansion_prompt,
    get_web_search_queries_prompt,
)


def get_concatenated_web_search_results_chain():
    """
    Get a chain that takes a list of summaries of web scraped pages and returns a concatenated string of the summaries.

    Input key is `query`.

    Returns a dict with keys `content` and `query`.
    """
    return (
        RunnablePassthrough.assign(query=lambda input_obj: input_obj["query"])
        | RunnablePassthrough.assign(
            urls=lambda input_obj: get_serp_links(input_obj["query"])
        )
        | (lambda x: [{"query": x["query"], "url": url} for url in x["urls"]])
        | get_scrape_and_summarize_webpage_chain().map()
        | (lambda x: {"content": "\n\n".join(x)})
    )


def get_scrape_and_summarize_webpage_chain():
    """
    Get a chain that takes a URL as input and returns a summary of the webpage.

    Input key is `url`.

    Returns a string with the URL and the summary of the webpage, or an empty string if no content.
    """
    return RunnablePassthrough.assign(
        summary=RunnablePassthrough.assign(
            content=lambda input_obj: scrape_webpage_text(input_obj["url"])
        )
        | get_summary_prompt()
        | get_llm("default", False)
        | StrOutputParser()
    ) | (
        lambda summarization_res: (
            f"INITIAL QUERY: {summarization_res['query']}\nSOURCE: {summarization_res['url']}\nSUMMARY: {summarization_res['summary']}"
            if summarization_res[
                "summary"
            ].strip()  # Only format if summary is non-empty
            else ""
        )
    )


def get_scrape_and_summarize_webpages_from_generated_queries_chain():
    """
    Get a chain that takes a query as an input and returns a concatenated string of summaries of the webpages.

    This works by:
    - generating search engine queries
    - scraping the first 3 web pages from the SERP
    - summarizing the web pages

    Input key is `query`.

    Returns a concatenated string of summaries of the web pages.
    """
    return (
        get_web_search_queries_chain()
        | (lambda input_obj: [{"query": q} for q in input_obj["queries"]])
        | RunnablePassthrough.assign(
            urls=lambda input_obj: get_serp_links(input_obj["query"])
        ).map()
        | (
            lambda x: [
                {"query": item["query"], "url": url}
                for item in x
                for url in item["urls"]
            ]
        )
        | get_scrape_and_summarize_webpage_chain().map()
        | (lambda summaries: "\n\n".join(summaries))
    )


def get_search_the_web_and_report_chain():
    """
    Get a chain that takes a user query in natural language as an input for a search on the web.

    Input key is `query`.

    Returns a written Markdown report (as a string) based on the web pages that were searched and scraped to be relevant to the query.
    """
    return (
        RunnablePassthrough.assign(
            query=get_user_query_expansion_prompt()
            | get_llm()
            | StrOutputParser()
            | json.loads
            | (lambda x: x["query"])
        )
        | RunnablePassthrough.assign(
            content=get_scrape_and_summarize_webpages_from_generated_queries_chain()
        )
        | get_summary_prompt()
        | get_llm("long-context", False)
        | StrOutputParser()
    )


def get_web_search_queries_chain():
    """
    Get a chain that outputs a list of 3 web search queries in JSON format from a user query written in natural language.

    Input key is `query`.
    """
    return get_web_search_queries_prompt() | get_llm() | StrOutputParser() | json.loads
