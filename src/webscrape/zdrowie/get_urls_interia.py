from selenium.webdriver import Keys, ActionChains

from src.utils.append_to_csv import append_to_csv
from src.utils.get_unique_filename import get_unique_filename
from selenium.webdriver.common.by import By
import time


def get_urls(csv_file: str, driver, row_limit: int | None = None, verbose: bool = False):
    csv_path = get_unique_filename(csv_file)
    url = "https://zdrowie.interia.pl/zdrowie"
    driver.get(url)

    category = 'zdrowie'

    csv_columns = ["headline", "category", "url"]

    time.sleep(2)

    collected_data = []
    scraped_urls = set()


    idx = 0

    while True:
        idx += 1

        if row_limit and len(collected_data) >= row_limit:
            print("Row limit reached. Exiting...")
            return collected_data

        try:
            articles = driver.find_elements(By.CLASS_NAME, "brief-list-item")

            for article in articles:
                anchor = article.find_element(By.TAG_NAME, "a")
                url = anchor.get_attribute("href")
                headline = article.find_element(By.TAG_NAME, "span").text

                if not headline or not url:
                    continue

                if url not in scraped_urls:
                    scraped_urls.add(url)

                    temp_data = {
                        "headline": headline,
                        "url": url,
                        "category": category,
                    }

                    collected_data.append(temp_data)
                    append_to_csv([temp_data], csv_path, csv_columns)

                    if verbose:
                        print(f"#{len(collected_data)} Collected: {headline} - {url}")

        except Exception as e:
            print(f"Error processing article: {e}")

        next_url = f"https://zdrowie.interia.pl/zdrowie,nPack,{idx}"
        driver.get(next_url)
        time.sleep(1)