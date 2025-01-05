import os
import time

from selenium.webdriver.common.by import By


def get_content(url, driver):
    try:
        content = ""
        driver.get(url)
        time.sleep(float(os.getenv("TIME_FOR_PAGE_LOAD")))

        article_body = driver.find_element(By.ID, "gazeta_article_body")

        for paragraph in article_body.find_elements(By.TAG_NAME, "p"):
            content += paragraph.text + " "

        return content.replace('"', '').strip()

    except Exception as e:
        return None