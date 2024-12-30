from langchain_openai import ChatOpenAI
import os
import sys
from typing import Literal
from yollama import get_llm as get_ollama_llm

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)


def get_llm(
    use_case: Literal["default", "long-context", "reasoning", "sql"] = "default",
    output_json: bool = True,
    provider: Literal["ollama", "openai"] = "ollama",
):
    if provider == "ollama":
        return get_ollama_llm(use_case, output_json)
    else:
        return get_openai_llm(use_case, output_json)


def get_openai_llm(
    use_case: Literal["default", "long-context", "reasoning", "sql"] = "default",
    output_json: bool = True,
):
    """
    Get a configured ChatOpenAI LLM instance with streaming and usage token output enabled.

    Currently, the use-case param is only passed for consistency with the yollama implementation.

    Returns:
        ChatOpenAI: Configured LLM instance
    """
    max_tokens = 16384

    model_name = "gpt-4o"
    # having trouble with `o1` like many other users
    # if use_case == "reasoning":
    #     model_name = "o1"

    if output_json:
        return ChatOpenAI(
            model_kwargs={"response_format": {"type": "json_object"}},
            model=model_name,
            max_tokens=max_tokens,
            streaming=True,
            stream_usage=True,
            temperature=0,
        )
    else:
        return ChatOpenAI(
            model=model_name,
            max_tokens=max_tokens,
            streaming=True,
            stream_usage=True,
            temperature=0,
        )


def log(message: str):
    print(f"\033[36m[DEBUG] \n{message}\n\033[0m")  # Cyan color for debug messages


def log_if_debug(message: str):
    if os.getenv("WITH_DEBUG_MESSAGES") == "true":
        log(message)
