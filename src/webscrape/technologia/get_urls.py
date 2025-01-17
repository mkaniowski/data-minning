from selenium.webdriver import Keys, ActionChains

from src.utils.append_to_csv import append_to_csv
from src.utils.get_unique_filename import get_unique_filename
from selenium.webdriver.common.by import By
import time


def get_urls(csv_file: str, driver, row_limit: int | None = None, verbose: bool = False):
    csv_path = get_unique_filename(csv_file)
    url = "https://wiadomosci.onet.pl/technologia"
    driver.get(url)

    category = 'technologia'

    csv_columns = ["headline", "category", "url"]

    time.sleep(2)

    collected_data = []
    scraped_urls = set()

    cookies = driver.find_element(By.CLASS_NAME, "cmp-intro_acceptAll")
    if cookies:
        ActionChains(driver).move_to_element(cookies).click().perform()
        time.sleep(2)

    while True:
        try:
            back_btn = driver.find_element(By.ID, "backBtn")
            if back_btn:
                ActionChains(driver).move_to_element(back_btn).click().perform()
                time.sleep(2)
        except Exception:
            print(f"No back button found or failed to click")

        articles = driver.find_elements(By.CLASS_NAME, "listItem")

        for article in articles:
            if row_limit and len(collected_data) >= row_limit:
                print("Row limit reached. Exiting...")
                return collected_data

            try:
                url = article.find_element(By.TAG_NAME, "a").get_attribute("href")
                headline = article.find_element(By.TAG_NAME, "h2").text

                if url not in scraped_urls:
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

        try:
            load_more_button = driver.find_element(By.XPATH, "//*[@id=\"articleDetailBottom\"]/article/div[2]/div/div[21]/span")
            ActionChains(driver).move_to_element(load_more_button).click().perform()
            time.sleep(2)
        except Exception as e:
            print(f"Error loading more content: {e}")