from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState
import os
import sys


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.lib import get_llm, log_if_debug
from agentic_search.prompts.web import get_web_search_agent_system_prompt
from agentic_search.tools.web import get_web_search_tools


def get_web_search_agent_node(state: MessagesState):
    """Get the agent node, which is the entry point for the agent."""
    # let's give our agent a personae
    sys_msg = SystemMessage(content=get_web_search_agent_system_prompt())
    # now bind tools to the agent
    llm_with_tools = get_llm("default", False).bind_tools(get_web_search_tools())
    log_if_debug(
        f"invoking web search agent with messages: {[sys_msg] + state['messages']}"
    )
    # using messages state keeps history during the graph execution (transient to one execution)
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
