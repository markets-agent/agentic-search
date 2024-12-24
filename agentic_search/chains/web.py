from langchain_core.output_parsers import StrOutputParser
import json
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.lib import get_llm
from agentic_search.prompts.web import (
    get_claim_verification_web_search_query_prompt,
    get_web_search_query_prompt,
    get_web_search_queries_prompt,
)


def get_claim_verification_web_search_query_chain():
    """
    Get a chain that outputs a single web search query in JSON format from a claim to verify.
    """
    return (
        get_claim_verification_web_search_query_prompt()
        | get_llm()
        | StrOutputParser()
        | json.loads
    )


def get_web_search_query_chain(excluded_queries: list[str] = []):
    """
    Get a chain that outputs a single web search query in JSON format from a user query written in natural language.

    Input key is `query`.
    """
    return get_web_search_query_prompt(excluded_queries) | get_llm() | StrOutputParser() | json.loads


def get_web_search_queries_chain(excluded_queries: list[str] = []):
    """
    Get a chain that outputs a list of x web search queries in JSON format from a user query written in natural language.

    Input key is `query`.
    """
    return get_web_search_queries_prompt(excluded_queries) | get_llm() | StrOutputParser() | json.loads
