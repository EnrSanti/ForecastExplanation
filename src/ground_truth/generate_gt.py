import json
import os

from .extract_pdf import text_extraction


def generate_gt(target_dates: list, output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    for date in target_dates:
        res = text_extraction(date)
        save_to_file(res, output_path, date)


def save_to_file(data: dict, output_path: str, date) -> None:
    # tmp
    out = os.path.join(output_path, date.strftime("%Y-%m-%d"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "gt.json"), "w") as file:
        file.write(json.dumps(data, indent=4))
