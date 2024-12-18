import os
import time

from selenium.webdriver.common.by import By


def get_content(url, driver):
    try:
        content = ""
        driver.get(url)
        time.sleep(float(os.getenv("TIME_FOR_PAGE_LOAD")))


        paragraphs = driver.find_elements(By.CLASS_NAME, "sc-bf72a589-0")

        for paragraph in paragraphs:
            content += paragraph.text + " "

        return content[:200].replace('"', '').strip()

    except Exception as e:
        return e