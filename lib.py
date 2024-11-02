from langchain_ollama import ChatOllama
from typing import Literal
import subprocess

default_llm_model_name = "llama3.1"
default_long_context_llm_model_name = "mistral-nemo"
default_sql_llm_model_name = "qwen2.5-coder"

def get_llm(use_case: Literal["default", "long-context", "sql"] = "default", output_json: bool = True):
    if use_case not in ["default", "long-context", "sql"]:
        raise ValueError("Invalid use case. Please choose from 'default', 'long-context', or 'sql'.")

    model_name = None
    num_ctx = None
    if use_case == "long-context":
        model_name = default_long_context_llm_model_name
        num_ctx = 32768
    elif use_case == "sql":
        model_name = default_sql_llm_model_name
        num_ctx = 16384
    else:
        model_name = default_llm_model_name
        num_ctx = 8192

    stdout, stderr = run_bash_command("ollama list")
    if stderr or model_name not in stdout:
        raise ValueError(f"Model {model_name} not found. Please install it with `ollama pull {model_name}`.")

    if output_json:
        return ChatOllama(
            format="json",
            model=model_name,
            # `num_ctx` is the size of the context window used to generate the next token
            num_ctx=num_ctx,
            # `num_predict` sets the maximum number of tokens to generate in the response,
            # -1 means infinite generation, -2 means fill context
            num_predict=-2,
            temperature=0,
        )
    else:
        return ChatOllama(
            model=model_name,
            num_ctx=num_ctx,
            num_predict=-2,
            temperature=0,
        )

def run_bash_command(command: str) -> tuple[str, str]:
    """Run a bash command and return its output and error streams.
    
    Args:
        command: The command to run
        
    Returns:
        tuple[str, str]: A tuple containing (stdout, stderr)
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    return stdout, stderr
