import csv
import os

from tqdm import tqdm

from src.utils.get_unique_filename import get_unique_filename


def process_urls(input_file, output_file, processor, driver):
    output_path = get_unique_filename(output_file)
    with open(input_file, mode="r", encoding="utf-8") as infile, open(output_path, mode="w", encoding="utf-8",
            newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ["headline", "category", "content", "url"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        count = 0

        for row in tqdm(reader):
            # os.system('cls' if os.name == 'nt' else 'clear')
            # print("Row:  ", count, end="\r")
            headline = row["headline"]
            category = row["category"]
            url = row["url"]

            content = processor(url, driver)

            if content is None:
                continue

            writer.writerow({"headline": headline, "category": category, "content": content, "url": url, })
            count = count + 1
