from bs4 import BeautifulSoup
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import requests


def get_serp_links(query: str, num_results: int = 3):
    ddg_search = DuckDuckGoSearchAPIWrapper()
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


def scrape_webpage_text(url: str):
    try:
        r = requests.get(url)
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
        # we return an empty string if there is an error so that we can continue the chain in which this function is used
        return ""
