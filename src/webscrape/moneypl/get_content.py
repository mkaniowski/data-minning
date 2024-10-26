import os
import time

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium.webdriver.common.by import By

load_dotenv()


def get_content(url, driver):
    # print("Scraping content for url: ", url)
    try:
        driver.get(url)
        time.sleep(float(os.getenv("TIME_FOR_PAGE_LOAD")))

        try:
            accept_button = driver.find_element(By.XPATH, "/html/body/div[3]/div/div[2]/div[3]/div/button[2]")
            accept_button.click()
        except Exception:
            pass
            # print(f"No cookie consent pop-up found or failed to click")

        soup = BeautifulSoup(driver.page_source, "html.parser")

        content_div = soup.find("div", class_="sc-hUMlYv fkDVoj")

        if content_div:
            paragraphs = content_div.find_all("p")
            content = " ".join([p.get_text() for p in paragraphs])
            return " ".join(content.split()[:200])
        else:
            return "Content not found"
    except Exception as e:
        return f"Error scraping URL: {e}"
