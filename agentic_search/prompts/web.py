from langchain_core.prompts import ChatPromptTemplate


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
    """
    web_search_queries_prompt_template = """Generate appropriate web search engine queries to find objective information about this user query:

---
{query}
---

IMPORTANT: 
- generate as many queries as needed to thoroughly cover the topic
- use fewer queries for simple topics, more for complex ones
- each query should target a distinct aspect of the information needed
- avoid redundant queries

Return JSON only:
{{"queries": ["query 1", "query 2", ...]}}"""
    return ChatPromptTemplate.from_template(web_search_queries_prompt_template)
