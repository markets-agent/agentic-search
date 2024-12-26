from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.functions.arxiv import format_search_results, get_arxiv_search_results
from agentic_search.lib import get_llm
from agentic_search.prompts.arxiv import get_arxiv_search_queries_prompt
from agentic_search.prompts.text import (
    get_formatted_report_prompt,
    get_qa_summary_prompt,
)


def get_arxiv_concatenated_summary_chain():
    return (
        # first generate the search queries
        get_arxiv_search_queries_chain()
        # transform the queries list into a list of query objects
        | (lambda x: [{"query": q} for q in x["queries"]])
        # map each query through the search and summarization chain
        | get_arxiv_search_results_summaries_chain().map()
        # format the results into a readable formatted string
        | (lambda x: format_search_results(x))
    )


def get_arxiv_final_report_chain():
    return (
        get_formatted_report_prompt()
        | get_llm(
            "default",
            False
        )
        | StrOutputParser()
    )


def get_arxiv_search_queries_chain():
    """
    Creates a chain that generates search queries for arXiv based on a user input query.

    Input key is `query`.

    Returns a chain that generates search queries.
    """
    return (
        get_arxiv_search_queries_prompt() | get_llm() | StrOutputParser() | json.loads
    )


def get_arxiv_search_results_summaries_chain():
    """
    Creates a chain that fetches and summarizes arXiv search results.

    Returns a chain that processes arXiv search results
    """
    # 1st level: the outer `RunnablePassthrough.assign` preserves all original input and adds a 'summary' field
    summarize_arxiv_search_result_partial = RunnablePassthrough.assign(
        summary=RunnablePassthrough.assign(
            # 2nd level: takes the 'result' field from the input and assigns it to 'content'
            content=lambda input_obj: input_obj["search_result"]
        )
        # this content is then passed to the next prompt
        | get_qa_summary_prompt()
        # we don't want a JSON output here, just the raw summary string
        | get_llm(False)
        | StrOutputParser()
    )

    # this returns a list of summaries, nested in objects having `query`, `search_result`, and `summary` fields
    return (
        # from the user input query, get the search results in Arxiv
        RunnablePassthrough.assign(
            search_results=lambda input: get_arxiv_search_results(input["query"])
        )
        | (
            # transform the results into a list of dictionaries,
            # each containing a query and a search result being a formatted string
            lambda list_of_results: [
                {"query": list_of_results["query"], "search_result": result}
                for result in list_of_results["search_results"]
            ]
        )
        # we then pass each query and result to the next prompt for summarization
        | summarize_arxiv_search_result_partial.map()
    )
