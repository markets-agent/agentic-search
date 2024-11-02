import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.chains.sql import (
    get_execute_sql_query_chain,
    get_generate_sql_query_chain,
    get_check_dql_chain,
    get_natural_language_response_from_sql_chain,
)


def get_natural_language_response_from_sql(query: str):
    sql_query = get_generate_sql_query_chain().invoke({"query": query})
    sql_is_valid = get_check_dql_chain().invoke({"query": sql_query})
    if not sql_is_valid["is_valid"]:
        return sql_is_valid["error_message"]

    sql_response = get_execute_sql_query_chain().invoke({"query": sql_query})

    return get_natural_language_response_from_sql_chain().invoke(
        {
            "query": query,
            "query": sql_query,
            "response": sql_response,
        }
    )
