from langchain_core.prompts import ChatPromptTemplate


def get_arxiv_search_queries_prompt():
    """
    Generates a prompt for generating search queries for arXiv based on a user input query.

    Returns a prompt for generating search queries.
    """
    arxiv_search_queries_prompt_messages = [
        (
            "user",
            """You are a research assistant.

Your job is to write 3 search queries to search Arxiv with a given user input query.
            
## GUIDELINES FOR QUERYING ARXIV

### searching by author:

- optimize name search by following "surname(s), forename(s) or surname(s), initial(s)" pattern: example `Hawking, S` or `Hawking, Stephen`
- for best results on multiple author names, separate individuals with a `;` (semicolon). Example: `Jin, D S; Ye, J`
- author names enclosed in quotes will return only exact matches; for example, `"Stephen Hawking"` will not return matches for `Stephen W. Hawking`
- diacritic character variants are automatically searched in the Author(s) field
- queries with no punctuation will treat each term independently

### wildcards:

- use `?` to replace a single character or `*` to replace any number of characters; this can be used in any field, but not in the first character position

### expressions:

- TeX expressions can be searched, enclosed in single `$` characters

### phrases:

- enclose phrases in double quotes for exact matches in title, abstract, and comments

### dates:

- sorting by announcement date will use the year and month the original version (v1) of the paper was announced
- sorting by submission date will use the year, month and day the latest version of the paper was submitted

### journal references:

- if a journal reference search contains a wildcard, matches will be made using wildcard matching as expected; for example, `math*` will match `math`, `maths`, `mathematics`, etc.
- if a journal reference search does not contain a wildcard, only exact phrases entered will be matched; for example, `math` would match `math` or `math` and science but not `maths` or `mathematics`
- all journal reference searches that do not contain a wildcard are literal searches: a search for `Physica A` will match all papers with journal references containing `Physica A`, but a search for `Physica A, 245 (1997) 181` will only return the paper with journal reference `Physica A, 245 (1997) 181`
       
Here is the user input query, delimited by dashes:
---
{query}
---

You must respond with a JSON containing a list of strings in the following format:
         
{{
    "queries": ["query 1", "query 2", "query 3"]
}}
         
You will ALWAYS include the original query as one of the queries.
The two other queries MUST be different from the original query.
DO NOT prefix a query with things like `title:`, `abstract:`, `author:`, etc. as this will limit the search to the specified field.""",
        )
    ]
    return ChatPromptTemplate.from_messages(arxiv_search_queries_prompt_messages)
