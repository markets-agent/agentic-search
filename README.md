# agentic search

Multimedia search capabilities for your projects, fully local.

## what is this ?

This repo holds the code for [a fully local agentic search tool package](https://pypi.org/project/agentic-search/), building up on this excellent [LangChain web search assistant YouTube tutorial](https://www.youtube.com/watch?v=DjuXACWYkkU). I want to automate my searches on various data sources, as well as their analysis.

It can search:

- a PostgreSQL database
- Arxiv papers summaries
- PDF and text documents (RAG)
- the web

The idea here is to modularize all that logic so that it can be reused in other projects.

## prerequisites

This has been tested on a Ubuntu 24 server. You'll need:

- at least 20GB of GPU VRAM for the best experience
- all the env vars listed in the `.env.example` file
  - be mindful that if `WITH_DEBUG_MESSAGES` is set to `true` (as in the `.env.example` file), the output to `stdout` will be more verbose
- Chrome and ChromeDriver installed
- Ollama installed
- the Ollama models you need to have installed are the ones specified in the [`yollama` package](https://pypi.org/project/yollama/)
- PostgreSQL installed and running (preferably with the `pgvector` extension installed, as the PostgreSQL generation code assumes you're using vectors in the prompts)
- Python 3 with `venv` and `pip` installed

## optional

- if `OPENAI_API_KEY` is set, the `web` capability will use OpenAI's models instead of Ollama's

## basic usage

`pip install agentic-search`

Let's say you want to test the `generate_sql_query` chain with `pytest` in your project.

```python
from agentic_search import get_generate_sql_query_chain

def test_get_generate_sql_query_chain():
    chain = get_generate_sql_query_chain()
    result = chain.invoke({"query": "what are the latest headlines"})
    assert isinstance(result, str)
    assert result.startswith("SELECT")
```

## how it works

### folder structure

```
agentic_search/
    capabilities/
    chains/
    functions/
    graphs/
    nodes/
    prompts/
```

We support **capabilities** for:

- arxiv
- PostgreSQL
- web

We support **chains** for:

- arxiv
- PostgreSQL
- text (RAG, works on PDFs as well)
- web

We support **graphs** for:

- web

### Functions

Functions are simple Python functions that can be used by other components of the system. They are located in the `functions/` folder.


### Prompts

Prompts are the LLM prompts used by the chains. They are located in the `prompts/` folder.

### Chains

Chains in this project are the basic building blocks that perform specific operations. They are located in the `chains/` folder and typically:

- Need to be explicitly invoked using the `.invoke()` method
- Are more granular and focused on a single responsibility
- Can be composed together to form more complex operations
- Return raw results that might need further processing

### Graphs

The chains can be orchestrated using graphs (located in the `graphs/` folder), which provides:

- Ability to create cyclical workflows (unlike traditional DAG-based solutions)
- State management between chain executions
- Built-in persistence for maintaining conversation context
- Support for human-in-the-loop interventions

The graphs themselves use nodes, which are steps during the graph execution that can be composed together at will, located in the `nodes/` folder.

Nodes can be plain agent nodes, with custom logic, or tools or set of tools (kinda like a toolbelt) that can be used by the agent.

### Capabilities

Capabilities (located in the `capabilities/` folder), on the other hand, are higher-level features that:

- Provide a more user-friendly interface by accepting natural language input
- Compose multiple chains together to create complete workflows
- Handle all the necessary orchestration between different chains
- Return processed, ready-to-use results (either as strings or JSON objects)
- Don't require explicit `.invoke()` calls from the end user, unlike chains and graphs

### Key Differences

1. **Abstraction Level**:
   - Chains: Low-level, single-purpose operations
   - Graphs: Infrastructure layer handling state and flow control
   - Capabilities: High-level, complete features that solve user problems

2. **Usage Pattern**:
   - Chains: Sequential units of work that involve LLMs and/or Python functions
   - Graphs: Manages the execution flow and state between components
   - Capabilities: Accept simple string inputs in natural language

3. **Composition**:
   - Chains: Are the building blocks that get composed together
   - Graphs: Provides the framework for connecting and orchestrating components
   - Capabilities: Are the composers that arrange chains and graphsinto useful features

4. **Output**:
   - Chains: May return raw or intermediate results
   - Graphs: Handles state updates and ensures proper data flow between nodes
   - Capabilities: Always return processed, user-friendly output

This architecture allows for better modularity and reusability, as chains and graphs can be mixed and matched to create different capabilities while keeping the core logic separate from the high-level features. The graphs layer ensures reliable execution and state management throughout the workflow. The capabilities layer, on the other hand, provide a user-friendly interface for interacting with the system.

## tests

To run the tests, follow these steps:

- Navigate to the project root directory.

- Run all tests:
   ```
   pytest
   ```

- To run tests from a specific file:
  ```
  pytest agentic_search/functions/test_text.py -vv -s
  ```

- To run a specific test function:
  ```
  pytest -k "test_normalize_and_lemmatize_text" agentic_search/functions/test_text.py -vv -s
  ```

## philosophy

### fully local models

I've decided to let go of OpenAI to:

- preserve my privacy
- save money

As local models are getting better and better, I'm confident that I'll be able to have a greater experience in the future, even with my small 20GB VRAM GPU. 🤔 Of course, it's not fast, but hey everything comes with tradeoffs right? 💪

### search logic

The Arxiv and web search features are heavily inspired by https://github.com/assafelovic/gpt-researcher, which basically splits up one search into multiple sub searches before generating a final output.

For instance, you would have:

- the user searching "what are the latest advancements in AI"
- an LLM generates a set of subqueries, like "latest advancements in NLP", "latest advancements in ML", etc.
- each subquery is sent to a search engine
- the results are aggregated and passed to another LLM which generates a final output

## contributions guidelines

🤝 I haven't made up my mind on contribution guidelines yet. I guess we'll update them as you contribute!