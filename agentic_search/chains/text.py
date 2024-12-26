import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import sys
from typing import Literal

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.functions.text import get_pdf_pages_docs
from agentic_search.lib import get_llm
from agentic_search.prompts.text import (
    get_claims_consistency_comparison_prompt,
    get_content_answers_to_query_prompt,
    get_content_summary_prompt,
    get_formatted_report_prompt,
    get_qa_summary_prompt,
)


def get_claims_consistency_comparison_chain():
    return (
        get_claims_consistency_comparison_prompt()
        | get_llm()
        | StrOutputParser()
        | json.loads
    )


def get_content_answers_to_query_chain():
    return (
        get_content_answers_to_query_prompt()
        | get_llm()
        | StrOutputParser()
        | json.loads
    )


def get_content_summary_chain():
    return get_content_summary_prompt() | get_llm() | StrOutputParser()


def get_pdf_report_chain():
    """
    Generates a report chain for a PDF document.

    Input key is `source`.
    """
    return (
        RunnablePassthrough.assign(
            results=lambda input: get_pdf_pages_docs(input["source"])
        )
        | (
            lambda pages: [
                {
                    "content": page.page_content,
                }
                for page in pages["results"]
            ]
        )
        | get_qa_summary_chain().map()
        | (
            lambda summaries: {
                "unstructured_text": "\n\n".join([f"""{s}""" for s in summaries])
            }
        )
        | RunnablePassthrough.assign(unstructured_text=lambda input: input)
        | get_formatted_report_prompt()
        | get_llm("long-context", False)
        | StrOutputParser()
    )


def get_qa_summary_chain(use_case: Literal["default", "long-context"] = "default"):
    """
    Generates a summary chain.

    Input keys are `content` and `query`.
    """
    return get_qa_summary_prompt(True) | get_llm(use_case, False) | StrOutputParser()
