from agentic_search.chains.web import get_search_the_web_and_report_chain
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState
from yollama import get_llm
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.lib import log_if_debug
from agentic_search.prompts.web import get_web_search_agent_system_prompt


def get_web_search_agent_node(state: MessagesState):
    """Get the agent node, which is the entry point for the agent."""
    # let's give our agent a personae
    sys_msg = SystemMessage(content=get_web_search_agent_system_prompt())
    # now bind tools to the agent
    llm_with_tools = get_llm("long-context", False).bind_tools(get_web_search_tools())
    log_if_debug(
        f"invoking web search agent with messages: {[sys_msg] + state['messages']}"
    )
    # using messages state keeps history during the graph execution (transient to one execution)
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


def get_web_search_tool_node(query: str, x: int):
    """Search the web for the given query and output a nicely formatted and readable Markdown document.

    1st argument: `query`
    2nd argument: `x` (the number of search engine queries to generate)

    Use this tool if you need to search the web for current information or information that is not in your knowledge base.
    """
    log_if_debug(f"invoking web search tool with query: {query}")
    return get_search_the_web_and_report_chain().invoke({"query": query, "x": x})


def get_web_search_tools():
    return [get_web_search_tool_node]
