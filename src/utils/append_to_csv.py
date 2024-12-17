import csv

def append_to_csv(data, csv_path, csv_columns):
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            if csvfile.tell() == 0:
                writer.writeheader()  # Write header only if file is empty
            writer.writerows(data)
    except IOError as e:
        print(f"I/O error({e.errno}): {e.strerror}")