from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from src.utils.append_to_csv import append_to_csv
from src.utils.get_unique_filename import get_unique_filename
import time

def get_urls(csv_file: str, driver, row_limit: int | None = None, verbose: bool = False):
    csv_path = get_unique_filename(csv_file)
    url = "https://kobieta.onet.pl/zdrowie"
    driver.get(url)

    csv_columns = ["headline", "category", "url"]
    time.sleep(2)

    collected_data = []
    scraped_urls = set()  # To track unique URLs

    while True:
        articles = driver.find_elements(By.CLASS_NAME, "itemBox")

        for article in articles:
            if row_limit and len(collected_data) >= row_limit:
                print("Row limit reached. Exiting...")
                return collected_data

            try:
                url = article.get_attribute("href")
                headline = article.find_element(By.TAG_NAME, "span").text
                category = "zdrowie"

                if headline == "MATERIAŁ PROMOCYJNY":
                    continue

                if url not in scraped_urls:  # Avoid duplicates
                    scraped_urls.add(url)

                    temp_data = {
                        "headline": headline,
                        "url": url,
                        "category": category
                    }

                    collected_data.append(temp_data)
                    append_to_csv([temp_data], csv_path, csv_columns)  # Append only new data to CSV

                    if verbose:
                        print(f"#{len(collected_data)} Collected: {headline} - {url}")

            except Exception as e:
                print(f"Error processing article: {e}")

        # Scroll down to load more content
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(1)

    return collected_data
