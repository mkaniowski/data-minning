import os
import time

from selenium.webdriver.common.by import By


def get_content(url, driver):
    try:
        content = ""
        driver.get(url)
        time.sleep(float(os.getenv("TIME_FOR_PAGE_LOAD")))

        try:
            lead_text = driver.find_element(By.ID, "lead")
        except:
            lead_text = ""

        paragraphs = driver.find_element(By.CLASS_NAME, "articleBody").find_elements(By.TAG_NAME, "p")

        for paragraph in paragraphs:
            content += paragraph.text + " "

        content = lead_text.text + " " + content

        return content[:200].replace('"', '').strip()

    except Exception as e:
        return None