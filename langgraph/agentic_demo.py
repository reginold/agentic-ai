# libraries
from dotenv import load_dotenv
import os
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import re
import json
from pygments import highlight, lexers, formatters

# load environment variables from .env file
_ = load_dotenv()

class TavilySearchClient:
    def __init__(self, api_key):
        self.client = TavilyClient(api_key=api_key)

    def search(self, query, include_answer=False, max_results=1):
        return self.client.search(query, include_answer=include_answer, max_results=max_results)


class DuckDuckGoSearchClient:
    def __init__(self):
        self.ddg = DDGS()

    def search(self, query, max_results=6):
        try:
            results = self.ddg.text(query, max_results=max_results)
            return [i["href"] for i in results]
        except Exception as e:
            print(f"returning previous results due to exception reaching ddg.")
            results = [ 
                "https://weather.com/weather/today/l/USCA0987:1:US",
                "https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8",
            ]
            return results  


class WebScraper:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def scrape(self, url):
        if not url:
            return "Weather information could not be found."
        
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            return "Failed to retrieve the webpage."

        soup = BeautifulSoup(response.text, 'html.parser')
        return soup


class WeatherInfoExtractor:
    def extract(self, soup):
        weather_data = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
            text = tag.get_text(" ", strip=True)
            weather_data.append(text)

        weather_data = "\n".join(weather_data)
        weather_data = re.sub(r'\s+', ' ', weather_data)
        return weather_data


class JsonFormatter:
    def format(self, json_data):
        parsed_json = json.loads(json_data.replace("'", '"'))
        formatted_json = json.dumps(parsed_json, indent=4)
        colorful_json = highlight(formatted_json,
                          lexers.JsonLexer(),
                          formatters.TerminalFormatter())
        return colorful_json


def main():
    # connect
    tavily_client = TavilySearchClient(api_key=os.environ.get("TAVILY_API_KEY"))
    ddg_client = DuckDuckGoSearchClient()
    web_scraper = WebScraper()
    weather_info_extractor = WeatherInfoExtractor()
    json_formatter = JsonFormatter()

    # run search
    result = tavily_client.search("What is in Nvidia's new Blackwell GPU?", include_answer=True)
    print(result["answer"])

    # choose location (try to change to your own city!)
    city = "San Francisco"
    query = f"""
        what is the current weather in {city}?
        Should I travel there today?
        "weather.com"
    """

    # search for weather info
    urls = ddg_client.search(query)
    for url in urls:
        print(url)

    # scrape first website
    url = urls[0]
    soup = web_scraper.scrape(url)

    # extract weather info
    weather_data = weather_info_extractor.extract(soup)
    print(f"Website: {url}\n\n")
    print(weather_data)

    # run search
    result = tavily_client.search(query, max_results=1)
    data = result["results"][0]["content"]
    print(data)

    # format json
    colorful_json = json_formatter.format(data)
    print(colorful_json)


if __name__ == "__main__":
    main()