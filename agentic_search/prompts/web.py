from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import os


def get_user_query_expansion_prompt() -> PromptTemplate:
    """
    Get prompt to expand user query in order to maximize the chances of finding relevant information on the web.

    Input and output key is "query".
    """
    user_query_expansion_prompt_template = """As a web search specialist, expand this query to optimize search relevance while maintaining its core intent:

Query: {query}

IMPORTANT:
1. The expanded query MUST directly relate to the original query's main topic
2. Do not add unrelated or tangential concepts
3. Focus on adding relevant context and synonyms

Return JSON: {{"query": "your expanded query"}}"""
    user_query_expansion_prompt = PromptTemplate.from_template(
        user_query_expansion_prompt_template
    )
    return user_query_expansion_prompt


def get_web_search_agent_system_prompt() -> str:
    prompt = """You are a precise research assistant with web search capabilities. Your tasks:
1. Provide accurate, current information
2. Synthesize multi-source information concisely
3. Include citations and maintain objectivity

Focus on authoritative, verifiable sources only.

Skip web search if you are completely certain of information.
Otherwise, perform targeted searches as needed."""
    return prompt


def get_web_search_queries_prompt() -> ChatPromptTemplate:
    """
    Get a prompt to generate a list of search engine queries from a user query.

    Input keys are:
    - `query`: the user query to expand
    - `x`: the number of queries to generate
    """
    web_search_queries_prompt_template = """Generate {x} web search engine queries to find objective information about this user query:

---
{query}
---

Return JSON only:
{{"queries": ["query 1", "query 2", "query 3"]}}"""
    return ChatPromptTemplate.from_template(web_search_queries_prompt_template)
