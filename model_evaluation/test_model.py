import json
import os
import sys
from PIL import Image
import pandas as pd


# Solve path problems
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from src.utils.teds import TEDS
from src.vocab.constant import CELL_SPECIAL
from src.utils import build_table_from_html_and_cell, html_table_template
from unitable_runner import UniTable


DATA_PATH = r"C:\Users\jmgarzonv\Desktop\EAFIT\Tesis\ground_truth\synthtabnet"
GROUND_TRUTH_FILE = "test_synthetic_data.jsonl"
RESULTS_PATH = r".\results"


def print_decoding_error(e):
    print(f"Error decoding JSON from line: {e.doc[:50]}...")
    print(f"Error message: {e}\n")

    # How many characters to show before and after
    context_range = 50  # Adjust this value as needed

    # Calculate start and end positions to slice
    start = max(0, e.pos - context_range)
    end = min(len(e.doc), e.pos + context_range)

    # Print the substring around the position of interest
    print(e.doc[start:end])
    print(" " * (e.pos - start) + "^")


def process_ground_truth_data(ground_truth_data):
    # Your code to process the ground truth data goes here
    anno_html_raw = ground_truth_data["html"]["structure"]["tokens"]
    anno_cell_raw = [
        "".join(cell["tokens"])
        for cell in ground_truth_data["html"]["cells"]
        if cell["tokens"]
    ]
    anno_html = []
    idx = 0
    while idx < len(anno_html_raw):
        if "[" in anno_html_raw[idx]:
            assert idx + 1 < len(anno_html_raw)
            assert anno_html_raw[idx + 1] == "]</td>"
            anno_html.append(anno_html_raw[idx] + "]</td>")
            idx = idx + 2
        else:
            anno_html.append(anno_html_raw[idx])
            idx = idx + 1

    anno_cell = []
    for txt in anno_cell_raw:
        for black in CELL_SPECIAL:
            txt = txt.replace(black, "")
        anno_cell.append(txt)

    anno_code = "".join(build_table_from_html_and_cell(anno_html, anno_cell))
    return anno_code


class TestModel:
    def __init__(self, image_path, ground_truth_data):
        self.image = Image.open(image_path).convert("RGB")
        self.anno_code = process_ground_truth_data(ground_truth_data)

    def inference(self, model):
        self.pred_code = model.predict(self.image)
        return self.pred_code

    def compute_metric(self):
        # Your code to compute the metric goes here
        # Evaluate table structure only (S-TEDS)
        metric = TEDS(structure_only=True)
        value = metric.evaluate(self.pred_code, html_table_template(self.anno_code))

        return value


def main():
    evaluate_ground_truth()


def evaluate_ground_truth():
    directories = os.listdir(DATA_PATH)
    model = UniTable()
    results = []
    for directory in directories:
        ground_truth_file = os.path.join(DATA_PATH, directory, GROUND_TRUTH_FILE)

        if not os.path.exists(ground_truth_file):
            print(f"File {ground_truth_file} does not exist")
            continue

        with open(ground_truth_file, "r", encoding="utf-8") as file:
            count = 0
            for line in file:
                count += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    ground_truth_data = json.loads(line)
                    image_path = os.path.join(
                        DATA_PATH,
                        directory,
                        "images",
                        ground_truth_data["split"],
                        ground_truth_data["filename"],
                    )
                    test = TestModel(image_path, ground_truth_data)
                    prediction = test.inference(model)
                    teds = test.compute_metric()
                    results.append(
                        {
                            "directory": directory,
                            "image_path": image_path,
                            "prediction": prediction,
                            "teds": teds,
                        }
                    )
                    # Execution report
                    print(f"Processed: Directory: {directory}: {count}, TEDS: {teds}")
                except json.JSONDecodeError as e:
                    print_decoding_error(e)
            print(f"{count} files processed for '{directory}'")
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"results_{directory}.csv", index=False)


if __name__ == "__main__":
    main()
