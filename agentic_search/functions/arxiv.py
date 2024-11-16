import arxiv


def format_search_results(search_results):
    """
    Formats search results into a readable string.

    The input search results are lists of dictionaries, each containing a query and a result.

    Returns a string with the formatted search results.
    """
    return "\n\n".join(
        [
            "\n\n".join(
                [
                    f"""---
ARXIV SEARCH QUERY:

{r['query']}

ARXIV SEARCH RESULT:

{r['search_result']}

ARXIV SEARCH RESULT SUMMARY:

{r['summary']}
---"""
                    for r in sr
                ]
            )
            for sr in search_results
        ]
    )


def get_arxiv_search_results(query: str, num_results: int = 3):
    """
    Get a list of arxiv papers summaries and their metadata formatted as strings.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        max_results=num_results, query=query, sort_by=arxiv.SortCriterion.Relevance
    )
    results_formatted = []
    for paper in client.results(search):
        authors_str = ", ".join([author.name for author in paper.authors])
        links_str = ", ".join([link.href for link in paper.links])
        published_str = paper.published.strftime("%B %d, %Y")
        results_formatted.append(
            {
                "authors": authors_str,
                "id": paper.get_short_id(),
                "links": links_str,
                "published": published_str,
                "summary": paper.summary,
                "title": paper.title,
            }
        )
    return [
        f"""Title: {result["title"]}
            
        ID: {result["id"]}
        
        Authors: {result["authors"]}

        Published: {result["published"]}

        Summary: {result["summary"]}

        Links: {result["links"]}
        
    """
        for result in results_formatted
    ]
