from bs4 import BeautifulSoup
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import os
import requests
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.lib import log_if_debug


def get_serp_links(query: str, num_results: int = 3):
    ddg_search = DuckDuckGoSearchAPIWrapper()
    results = ddg_search.results(query, num_results)
    log_if_debug(f"serp results for query {query}: {results}")
    return [r["link"] for r in results]


def scrape_webpage_text(url: str, timeout: int = 5):
    log_if_debug(f"scraping webpage {url}")
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            # BeautifulSoup transforms a complex HTML document into a tree of Python objects,
            # such as tags, navigable strings, or comments
            soup = BeautifulSoup(r.text, "html.parser")
            # separating all extracted text with a space
            text = soup.get_text(separator=" ", strip=True)
            return text
        else:
            raise Exception(f"failed to scrape webpage with status: {r.status_code}")
    except Exception as e:
        log_if_debug(f"error scraping webpage {url}: {e}")
        # we return an empty string if there is an error so that we can continue the chain in which this function is used
        return ""
