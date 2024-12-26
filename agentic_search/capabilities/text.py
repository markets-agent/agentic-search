from langchain_core.messages import HumanMessage
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.chains.text import get_content_summary_chain


def get_running_summary(items: list[str]):
    """
    Get a running summary from a list of items.


    With a list of items, returns a written Markdown report of the running summary.
    """
    summary = ""
    while len(items) > 0:
        current_item = items.pop(0)
        if summary == "":
            summary = current_item
        else:
            summary = get_content_summary_chain().invoke({
                "content": summary + "\n\n" + current_item,
            })
    
    return {
        "summary": summary
    }