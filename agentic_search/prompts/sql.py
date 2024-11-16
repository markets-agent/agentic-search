from langchain_core.prompts import ChatPromptTemplate


def get_check_dql_prompt():
    check_dql_prompt_template = """You are a SQL expert and PostgreSQL DBA. 
You are a given a SQL query, your job is:

- to check that it's syntactically correct
- to check that it's only doing data retrieval (i.e. only SELECT statements)

Here is the SQL query, delimited by dashes:
---
{query}
---

You are to output a JSON object with the following fields:
- is_valid: a boolean indicating whether the query is syntactically correct and only does data retrieval
- error_message: an error message if the query is not syntactically correct, otherwise null"""
    return ChatPromptTemplate.from_template(check_dql_prompt_template)


def get_sql_query_prompt():
    sql_query_prompt_template = """You are a SQL expert and PostgreSQL DBA. 
Write an optimized PostgreSQL query that accurately answers the user's question while maximizing performance and maintainability. The query must be DQL only (SELECT statements only, no DDL/DML operations).

Schema:
---
{db_schema}
---

User Question:
---
{query}
---

Guidelines:
- Ensure the query precisely answers the user's question
- Only use SELECT statements (no CREATE, INSERT, UPDATE, DELETE, etc.)
- Use meaningful table aliases that reflect the table name (e.g., 'headlines' for 'investment_news_headlines')
- Use vector similarity search for vector fields when appropriate
- Include performance-focused SQL comments
- Ensure proper indexing hints where beneficial
- Consider query plan efficiency
- Always qualify column references with table aliases
- Format joins with consistent indentation
- Verify all required data is included in the result set
- Use consistent naming conventions throughout the query

Example alias format:
- investment_news_headlines AS headlines
- investment_news_dynamic_categories AS categories
- investment_news_headlines_dynamic_categories AS headline_categories

Important: Output the raw SQL query only, without any markdown formatting, SQL tags, or other delimiters. Do not wrap the query in ```sql``` tags or any other formatting."""
    return ChatPromptTemplate.from_template(sql_query_prompt_template)


def get_sql_to_natural_language_response_prompt():
    sql_to_natural_language_response_prompt_template = """As a SQL expert DBA and communicator, translate the following database query results into natural language.

Query: 
{query}

Schema:
{db_schema}

SQL:
{query}

Results:
{response}

Provide a clear, human-readable response based solely on the above inputs. Do not include technical details or SQL syntax in your explanation."""
    return ChatPromptTemplate.from_template(
        sql_to_natural_language_response_prompt_template
    )
