import os
import time

def get_content(url, driver):
    try:
        driver.get(url)
        time.sleep(float(os.getenv("TIME_FOR_PAGE_LOAD")))

    except Exception as e:
        return f"Error scraping URL: {e}"