from selenium.webdriver import Keys, ActionChains

from src.utils.append_to_csv import append_to_csv
from src.utils.get_unique_filename import get_unique_filename
from selenium.webdriver.common.by import By
import time


def get_urls(csv_file: str, driver, row_limit: int | None = None, verbose: bool = False):
    csv_path = get_unique_filename(csv_file)
    url = "https://www.sport.pl/pilka/0,0.html"
    driver.get(url)

    category = 'sport'

    csv_columns = ["headline", "category", "url"]

    time.sleep(2)

    collected_data = []
    scraped_urls = set()

    # Accept cookies
    try:
        cookies_btn = driver.find_element(By.CLASS_NAME, "onetrust-accept-btn-handler")
        ActionChains(driver).move_to_element(cookies_btn.find_element(By.TAG_NAME, "span")).click().perform()
        time.sleep(1)
    except Exception:
        print("No cookies found or failed to click")

    idx = 0

    while True:
        idx += 1

        if row_limit and len(collected_data) >= row_limit:
            print("Row limit reached. Exiting...")
            return collected_data

        try:
            articles = driver.find_elements(By.CLASS_NAME, "article")

            for article in articles:
                anchor = article.find_element(By.TAG_NAME, "a")
                url = anchor.get_attribute("href")
                headline = anchor.get_attribute("title")

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

        next_url = f"https://www.sport.pl/pilka/0,0.html?str={idx}_26935331"
        driver.get(next_url)
        time.sleep(1)