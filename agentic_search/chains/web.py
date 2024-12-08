from langchain_core.output_parsers import StrOutputParser
import json
import os
import sys
from yollama import get_llm

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.prompts.web import (
    get_web_search_queries_prompt,
)

def get_web_search_queries_chain():
    """
    Get a chain that outputs a list of x web search queries in JSON format from a user query written in natural language.

    Input key is `query`.
    """
    return get_web_search_queries_prompt() | get_llm() | StrOutputParser() | json.loads
