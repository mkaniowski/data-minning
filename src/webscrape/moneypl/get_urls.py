# %%
import os
import time

from dotenv import load_dotenv
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

from src.utils.append_to_csv import append_to_csv
from src.utils.get_unique_filename import get_unique_filename

load_dotenv()


# %%
def get_urls(csv_file: str, driver, row_limit: int | None = None, verbose: bool = False):
    csv_path = get_unique_filename(csv_file)
    url = "https://www.money.pl/sekcja/prawo/"
    driver.get(url)

    csv_columns = ["headline", "category", "url"]

    collected_data = []
    button_click_count = 0
    last_row_count = 0

    loop_check = True

    if row_limit is not None:
        loop_check = len(collected_data) < row_limit

    try:
        accept_button = driver.find_element(By.XPATH, "/html/body/div[3]/div/div[2]/div[3]/div/button[2]", )
        accept_button.click()
    except Exception:
        print(f"No cookie consent pop-up found or failed to click")

    while loop_check:
        if verbose: print(f"Page Source Loaded, length: {len(driver.page_source)}")

        headlines = driver.find_elements(By.CSS_SELECTOR, "h3.sc-2gaihw-9.jAoADu")
        new_headlines = headlines[last_row_count:]
        if verbose: print(f"New headlines found: {len(new_headlines)}")

        temp_data = []

        for headline_element in new_headlines:
            headline = headline_element.text

            anchor = headline_element.find_element(By.TAG_NAME, "a")
            url = anchor.get_attribute("href")
            category = driver.execute_script(
                "return window.getComputedStyle(arguments[0], '::before').getPropertyValue('content');", anchor)

            category = category.strip('"') if category else "Unknown"

            if verbose: print(f"Headline: {headline}, Category: {category}, Url: {url}")

            if headline and category:
                temp_data.append({"headline": headline, "category": category, "url": url})

            if row_limit is not None and len(collected_data) + len(temp_data) >= row_limit:
                break

        collected_data.extend(temp_data)

        append_to_csv(temp_data, csv_path, csv_columns)
        if verbose:
            print(f"Rows collected so far: {len(collected_data)}")
        else:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Rows collected so far: {len(collected_data)}", end="\r")

        last_row_count = len(headlines)

        if row_limit is not None and len(collected_data) >= row_limit:
            break

        try:
            load_more_button = driver.find_element(By.CLASS_NAME, "sc-143eo0d-0.sc-1j5p3rt-0.jGLyum")
            ActionChains(driver).move_to_element(load_more_button).click().perform()

            button_click_count += 1
            if verbose: print(f"'Load More' button clicked {button_click_count} times")

            time.sleep(float(os.getenv("TIME_BETWEEN_LOAD_MORE")))
        except Exception as e:
            print("No more load button found or an error occurred:", str(e))
            break

    print(f"Scraping completed. {len(collected_data)} rows collected and saved to {csv_path}")

    driver.quit()
