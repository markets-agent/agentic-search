import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import sys
from ypostgres_lib import run_static_dql

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.functions.sql import get_postgres_db_schema
from agentic_search.lib import get_llm
from agentic_search.prompts.sql import (
    get_check_dql_prompt,
    get_sql_query_prompt,
    get_sql_to_natural_language_response_prompt,
)


def get_check_dql_chain():
    return (
        RunnablePassthrough.assign(query=lambda input_obj: input_obj["query"])
        | get_check_dql_prompt()
        | get_llm()
        | StrOutputParser()
        | json.loads
    )


def get_execute_sql_query_chain():
    return (
        RunnablePassthrough.assign(query=lambda input_obj: input_obj["query"])
        | (lambda input_obj: run_static_dql(input_obj["query"]))
    )


def get_generate_sql_query_chain():
    """
    Generate a SQL query from a natural language query.
    
    The input key you need to provide is "query".

    Returns a raw SQL query (to be checked later during processing).
    """
    return (
        RunnablePassthrough.assign(db_schema=lambda _: get_postgres_db_schema())
        | get_sql_query_prompt()
        | get_llm("sql", False)
        | StrOutputParser()
    )


def get_natural_language_response_from_sql_chain():
    return (
        RunnablePassthrough.assign(db_schema=lambda _: get_postgres_db_schema())
        | get_sql_to_natural_language_response_prompt()
        | get_llm("long-context", False)
        | StrOutputParser()
    )
