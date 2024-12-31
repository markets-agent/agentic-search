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
    """
    Get a configured LLM instance based on the specified provider.

    Args:
        use_case: The use case for the LLM. Options are "default", "long-context", "reasoning", or "sql".
        output_json: Whether the LLM should output JSON format responses.
        provider: The LLM provider to use. Options are "ollama" or "openai".

    Returns:
        A configured LLM instance from either OpenAI or Ollama based on the provider parameter.
        This instance will be `ollama` by default if provider is not expected.
    """
    if provider == "openai":
        return get_openai_llm(use_case, output_json)
    else:
        return get_ollama_llm(use_case, output_json)


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

    model_name = "gpt-4o-mini"
    # having trouble with `o1` like many other users
    if use_case == "reasoning":
        model_name = "gpt-4o"

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


def get_websearch_llm():
    return get_llm("default", False, get_websearch_llm_provider())


def get_websearch_llm_provider():
    return os.getenv("WEBSEARCH_LLM_PROVIDER", "ollama")


def log(message: str):
    print(f"\033[36m[DEBUG] \n{message}\n\033[0m")  # Cyan color for debug messages


def log_if_debug(message: str):
    if os.getenv("WITH_DEBUG_MESSAGES") == "true":
        log(message)
