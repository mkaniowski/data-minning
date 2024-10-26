import csv


def append_to_csv(data, csv_file: str, csv_columns: list[str]):
    with open(csv_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        if csvfile.tell() == 0:
            writer.writeheader()
        for row in data:
            writer.writerow(row)
