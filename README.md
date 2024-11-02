# agentic search

## what is this ?

This repo holds the code for [an agentic search tool package](https://pypi.org/project/agentic-search/), building up on this excellent [LangChain web search assistant YouTube tutorial](https://www.youtube.com/watch?v=DjuXACWYkkU). 

It can search:

- a PostgreSQL database
- Arxiv papers summaries
- PDF and text documents (RAG)
- the web

The idea here is to modularize all that logic so that it can be reused in other projects.

## prerequisites

This has been tested on a Ubuntu 24 server. You'll need:

- all the env vars listed in the `.env.example` file
- Ollama installed
- the Ollama models you'll need to install are:

```python
default_llm_model_name = "llama3.1"
default_long_context_llm_model_name = "mistral-nemo"
default_sql_llm_model_name = "qwen2.5-coder"
```

- PostgreSQL installed and running (preferably with the `pgvector` extension installed, as the SQL generation code assumes you're using vectors in the prompts)
- Python 3 with `venv` and `pip` installed

## how it works

## Capabilities vs Chains

### Chains

Chains in this project are the basic building blocks that perform specific operations. They are located in the `chains/` folder and typically:

- Need to be explicitly invoked using the `.invoke()` method
- Are more granular and focused on a single responsibility
- Can be composed together to form more complex operations
- Return raw results that might need further processing

### Capabilities

Capabilities (located in the `capabilities/` folder), on the other hand, are higher-level features that:

- Provide a more user-friendly interface by accepting natural language input
- Compose multiple chains together to create complete workflows
- Handle all the necessary orchestration between different chains
- Return processed, ready-to-use results (either as strings or JSON objects)
- Don't require explicit `.invoke()` calls from the end user

### Key Differences

1. **Abstraction Level**:
   - Chains: Low-level, single-purpose operations
   - Capabilities: High-level, complete features that solve user problems

2. **Usage Pattern**:
   - Chains: Require explicit `.invoke()` calls and knowledge of input structure
   - Capabilities: Accept simple string inputs in natural language

3. **Composition**:
   - Chains: Are the building blocks that get composed together
   - Capabilities: Are the composers that arrange chains into useful features

4. **Output**:
   - Chains: May return raw or intermediate results
   - Capabilities: Always return processed, user-friendly output

This architecture allows for better modularity and reusability, as chains can be mixed and matched to create different capabilities while keeping the core logic separate from the high-level features.

## how to run a demo

- `cp .env.example .env` (and fill in the blanks)
- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- touch `demo.py` and copy the folowing contents:

```python
#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from langserve import add_routes
import uvicorn

# ! of course you'll need to add the relevant dependencies to the `requirements.txt` file

from chains import (
    generate_arxiv_search_report,
    generate_sql_nlp_response,
    generate_pdf_report, 
    generate_web_search_report
)

app = FastAPI(
  title="agentic search api",
  version="1.0",
  description="a web API that serves an AI-powered agentic search tool that can generate reports by crawling various sources for relevant information and summarizing it",
)
add_routes(
    app,
    generate_arxiv_search_report,
    path="/generate-arxiv-search-report",
)
add_routes(
    app,
    generate_pdf_report,
    path="/generate-pdf-report",
)
add_routes(
    app,
    generate_web_search_report,
    path="/generate-web-search-report",
)
add_routes(
    app,
    generate_sql_nlp_response,
    path="/sql-qa",
)
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
```

- `python3 demo.py`
- go to `localhost:8000` and search for something:
    - in a pdf by going to `/generate-pdf-report/playground/` route:
        - enter the absolute file path for the document you want to summarize
    - in Arxiv by going to `/generate-arxiv-search-report/playground/` route
    - on the web by going to `/generate-web-search-report/playground/` route

I've chosen to keep the demo out of the project to preserve the modularity of the code and to reduce the number of dependencies to install.

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