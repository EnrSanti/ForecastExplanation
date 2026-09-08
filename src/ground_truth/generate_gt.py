import json
import os
from datetime import datetime

from extract_pdf import text_extraction


def generate_gt(target_dates, output_path):
    output_path = os.path.join(output_path, "gt")
    os.makedirs(output_path, exist_ok=True)
    for date in target_dates:
        res = text_extraction(date)
        save_to_file(res, output_path, date)


def save_to_file(data, output_path, date):
    # tmp
    out = os.path.join(output_path, date.strftime("%Y-%m-%d"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "gt.json"), "w") as file:
        file.write(json.dumps(data))


if __name__ == "__main__":
    target_date = datetime(2019, 1, 1)
    generate_gt([target_date], "tmp")
