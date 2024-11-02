import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.chains.arxiv import (
    get_arxiv_concatenated_summary_chain,
    get_arxiv_final_report_chain,
)


def generate_arxiv_search_report(query: str):
    """
    Generates a final report from an arXiv search query.
    """
    concatenated_summary = get_arxiv_concatenated_summary_chain().invoke(
        {"query": query}
    )

    return get_arxiv_final_report_chain().invoke(
        {"unstructured_text": concatenated_summary}
    )
