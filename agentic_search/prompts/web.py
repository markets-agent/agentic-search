from langchain_core.prompts import ChatPromptTemplate


def get_web_search_queries_prompt():
    web_search_queries_prompt_template = """Generate 3 search engine queries to find objective information about:
---
{query}
---

Return JSON only:
{{"queries": ["query 1", "query 2", "query 3"]}}"""
    return ChatPromptTemplate.from_template(web_search_queries_prompt_template)
