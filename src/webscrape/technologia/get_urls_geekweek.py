from selenium.webdriver import Keys, ActionChains

from src.utils.append_to_csv import append_to_csv
from src.utils.get_unique_filename import get_unique_filename
from selenium.webdriver.common.by import By
import time


def get_urls(csv_file: str, driver, row_limit: int | None = None, verbose: bool = False):
    csv_path = get_unique_filename(csv_file)
    url = "https://geekweek.interia.pl/technologia"
    driver.get(url)

    category = 'technologia'

    csv_columns = ["headline", "category", "url", "sub_category"]

    time.sleep(2)

    collected_data = []
    scraped_urls = set()

    try:
        rodo_btn = driver.find_element(By.CLASS_NAME, "rodo-popup-agree")
        ActionChains(driver).move_to_element(rodo_btn).click().perform()
        time.sleep(1)
    except Exception:
        print(f"No RODO pop-up found or failed to click")

    idx = 0

    while True:
        idx += 1

        articles = driver.find_elements(By.CLASS_NAME, "brief-list-item")

        for article in articles:
            if row_limit and len(collected_data) >= row_limit:
                print("Row limit reached")
                return collected_data

            try:
                url = article.find_element(By.TAG_NAME, "a").get_attribute("href")
                headline = article.find_element(By.TAG_NAME, "h2").text
                desc = article.find_element(By.CLASS_NAME, "ids-card__description-container")
                sub_category = desc.find_element(By.TAG_NAME, "span").text

                if url not in scraped_urls:
                    scraped_urls.add(url)

                    temp_data = {
                        "headline": headline,
                        "url": url,
                        "category": category,
                        "sub_category": sub_category
                    }

                    collected_data.append(temp_data)
                    append_to_csv([temp_data], csv_path, csv_columns)

                    if verbose:
                        print(f"#{len(collected_data)} Collected: {headline} - {url}")

            except Exception as e:
                print(f"Error processing article: {e}")

        next_url = f"https://geekweek.interia.pl/technologia,nPack,{idx}"
        driver.get(next_url)
        time.sleep(1)