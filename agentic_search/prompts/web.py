from langchain_core.prompts import ChatPromptTemplate


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
