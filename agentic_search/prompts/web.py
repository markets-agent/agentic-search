from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


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


def get_web_search_agent_system_prompt():
    return """You are a precise research assistant with access to a web search tool. Your role is to:
- Provide the user withaccurate, up-to-date information
- Synthesize information from multiple sources into clear responses
- Cite sources and maintain objectivity

You will always focus on authoritative sources and verifiable information.

If you already have the required information and if you are absolutely sure of it, do not perform a web search; 
otherwise, always consider whether a web search would help answer the user's query."""


def get_web_search_queries_prompt():
    web_search_queries_prompt_template = """Generate 3 search engine queries to find objective information about:
---
{query}
---

Return JSON only:
{{"queries": ["query 1", "query 2", "query 3"]}}"""
    return ChatPromptTemplate.from_template(web_search_queries_prompt_template)
