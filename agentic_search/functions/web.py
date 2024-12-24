from aiocache import Cache
import asyncio
from asyncio import Semaphore
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException
import json
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import sys
from typing import List

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from agentic_search.lib import log_if_debug

# sharing only one Chrome instance across all requests
chrome_semaphore = Semaphore(1)

class CachedDDGWrapper:
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
            ddg_wrapper = DDGS(proxy=None)
            results = ddg_wrapper.text(query, max_results=self.max_results)

            if not results:
                raise DuckDuckGoSearchException("No results returned")

            await self.cache.set(query, json.dumps(results))
            return results

        except (DuckDuckGoSearchException, RatelimitException, Exception) as e:
            log_if_debug(f"Search attempt failed: {str(e)}")
            return []


def get_chrome_instance():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.binary_location = os.getenv(
        "CHROME_BINARY_LOCATION", "/usr/bin/google-chrome"
    )
    chrome_options.page_load_strategy = "eager"

    # the Chrome driver is the "bridge" between Python code and the Chrome browser;
    # the service starts and manages the chrome driver binary that actually controls Chrome,
    # it also handles the communication between Python and the driver,
    # the provided timeout is the timeout for the service to start the driver
    chrome_instance = webdriver.Chrome(
        options=chrome_options,
        service=webdriver.ChromeService(timeout=2),
    )
    return chrome_instance


async def get_serp_links(query: str, num_results: int = 3):
    ddg_search = CachedDDGWrapper(max_results=num_results)
    results = await ddg_search.results(query)
    log_if_debug(f"serp results for query {query}: {results}")
    return results


def get_webpage_soup(
    webpage_url: str, chrome_instance: webdriver.Chrome, timeout: int = 4
) -> BeautifulSoup:
    try:
        wait = WebDriverWait(chrome_instance, timeout)
        chrome_instance.get(webpage_url)

        # check if browser is still responsive
        try:
            chrome_instance.current_url  # this will throw if browser crashed
        except:
            # clean up crashed instance
            chrome_instance.quit() 
            raise Exception("browser became unresponsive")


        wait.until(
            lambda d: d.execute_script('return document.readyState') == 'complete',
        )
        return BeautifulSoup(chrome_instance.page_source, "html.parser")
    except Exception as e:
        log_if_debug(f"`get_webpage_soup` => error getting webpage soup for {webpage_url}: {e}")
        return None


def get_webpage_soup_text(
    webpage_url: str, chrome_instance: webdriver.Chrome, timeout: int = 4
) -> str:
    soup = None
    try:
        soup = get_webpage_soup(webpage_url, chrome_instance, timeout)
        if soup is None:
            return ""
        text = soup.get_text(separator=" ", strip=True)
        text += f"\n\nSOURCE: {webpage_url}\n\n"
        return text
    except Exception as e:
        log_if_debug(f"`get_webpage_soup_text` => error getting webpage soup for {webpage_url}: {e}")
        return ""

async def get_webpage_soup_text_async(webpage_url: str, timeout: int = 4) -> str:
    async with chrome_semaphore:  # wait for available slot
        # create a new Chrome instance for each request to avoid thread safety issues
        chrome_instance = get_chrome_instance()
        try:
            # run the synchronous selenium operations in a thread pool
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                text = await loop.run_in_executor(
                    pool,
                    lambda: get_webpage_soup_text(webpage_url, chrome_instance, timeout),
                )
                # always close the Chrome instance
                chrome_instance.quit()
                return text
        except Exception as e:
            log_if_debug(f"`get_webpage_soup_text_async` => error getting webpage soup for {webpage_url}: {e}")
            chrome_instance.quit()
            return ""

async def get_webpages_soups_text_async(
    urls: List[str], timeout_for_page_load: int = 3
) -> List[str]:
    # create tasks for all URLs
    tasks = [get_webpage_soup_text_async(url, timeout_for_page_load) for url in urls]

    # gather all results,
    # using `return_exceptions=True` to return exceptions as well;
    # this allows to not stop the scraping process when one page fails to load
    soups_text = await asyncio.gather(*tasks, return_exceptions=True)

    # filter out exceptions and combine text
    content = []
    for text in soups_text:
        if isinstance(text, Exception):
            log_if_debug(f"error getting webpages soups text: {text}")
            continue
        if text:  # Check if text is not None
            content.append(text)

    return content
