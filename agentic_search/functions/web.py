from aiocache import Cache
from bs4 import BeautifulSoup
from duckduckgo_search import AsyncDDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException
import json
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import sys
from typing import List

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.lib import log_if_debug


class ThrottledDDGWrapper:
    def __init__(self, max_results: int):
        use_redis = os.getenv("USE_REDIS", "false").lower() == "true"
        if use_redis:
            self.cache = Cache(Cache.REDIS, ttl=300)  # 5-minute redis cache
        else:
            self.cache = Cache(Cache.MEMORY, ttl=300)  # 5-minute in memory cache
        self.max_results = max_results

    async def results(self, query):
        # check cache first
        cached_result = await self.cache.get(query)
        if cached_result:
            return json.loads(cached_result)

        # simplified retry logic - only one retry
        ddg_wrapper = None
        try:
            ddg_wrapper = AsyncDDGS(proxy=None)
            results = await ddg_wrapper.atext(query, max_results=self.max_results)

            if not results:
                raise DuckDuckGoSearchException("No results returned")

            await self.cache.set(query, json.dumps(results))
            return results

        except (DuckDuckGoSearchException, RatelimitException, Exception) as e:
            log_if_debug(f"Search attempt failed: {str(e)}")
            return []


def get_chrome_instance(timeout_for_page_load: int = 4):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.binary_location = os.getenv(
        "CHROME_BINARY_LOCATION", "/usr/bin/google-chrome"
    )
    chrome_options.page_load_strategy = "eager"

    chrome_instance = webdriver.Chrome(
        options=chrome_options,
        service=webdriver.ChromeService(timeout=timeout_for_page_load),
    )
    return chrome_instance


async def get_serp_links(query: str, num_results: int = 3):
    ddg_search = ThrottledDDGWrapper(max_results=num_results)
    results = await ddg_search.results(query)
    log_if_debug(f"serp results for query {query}: {results}")
    return results


def get_webpage_soup(
    webpage_url: str, chrome_instance: webdriver.Chrome, timeout: int = 4
) -> BeautifulSoup:
    soup = None
    try:
        chrome_instance.set_page_load_timeout(timeout)
        chrome_instance.get(webpage_url)
        soup = BeautifulSoup(chrome_instance.page_source, "html.parser")
    except Exception as e:
        log_if_debug(f"error getting webpage soup for {webpage_url}: {e}")
    return soup


def get_webpage_soup_text(
    webpage_url: str, chrome_instance: webdriver.Chrome, timeout: int = 4
) -> BeautifulSoup:
    soup = None
    try:
        soup = get_webpage_soup(webpage_url, chrome_instance, timeout)
        text = soup.get_text(separator=" ", strip=True)
        text += f"\n\nSOURCE: {webpage_url}\n\n"
    except Exception as e:
        log_if_debug(f"error getting webpage soup for {webpage_url}: {e}")
    return text


def get_webpages_soups_text(urls: List[str], timeout_for_page_load: int = 4) -> str:
    chrome_instance = get_chrome_instance(timeout_for_page_load)

    soups_text = []
    for url in urls:
        try:
            soups_text.append(get_webpage_soup_text(url, chrome_instance))
        except Exception as e:
            log_if_debug(f"error getting webpages soups text: {e}")
            continue

    chrome_instance.quit()

    # extract text from each soup
    content = ""
    for text in soups_text:
        content += text
        content += "\n\n---\n\n"

    return content
