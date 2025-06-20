import os
import json
import re
from collections import defaultdict

# Adjust these paths
input_dir = "/ocr_output/"
output_dir = "merged_outputs/"
os.makedirs(output_dir, exist_ok=True)

# Helper function to load JSON safely
def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Group JSON files by original batch document name
batch_files = defaultdict(list)

# Walk through input directories and organize files
for root, _, files in os.walk(input_dir):
    for fname in sorted(files):
        if fname.endswith(".json"):
            # Extract original batch name, e.g., "minneapolis_1910_batch_1"
            match = re.match(r"(minneapolis_\d+_batch_\d+)-\d+\.json", fname)
            if match:
                batch_name = match.group(1)
                full_path = os.path.join(root, fname)
                batch_files[batch_name].append(full_path)

# Merge pages for each batch
for batch_name, json_files in batch_files.items():
    merged_document = {
        "pages": [],
        "text": "",
        "entities": [],
        "batch_name": batch_name
    }
    cumulative_text_length = 0  # Keep track of text offset across pages

    # Sort JSON files to maintain correct page order
    json_files.sort(key=lambda x: int(re.search(r'-(\d+)\.json$', x).group(1)))

    for json_file in json_files:
        page_data = load_json(json_file)

        # Merge text
        page_text = page_data.get("text", "")
        merged_document["text"] += page_text + "\n"  # separate pages by newline

        # Adjust page numbers if necessary
        for page in page_data.get("pages", []):
            merged_document["pages"].append(page)

        # Merge entities (if any)
        for entity in page_data.get("entities", []):
            # Adjust entity text offsets for cumulative text length
            text_anchor = entity.get("textAnchor", {}).get("textSegments", [])
            for segment in text_anchor:
                if "startIndex" in segment:
                    segment["startIndex"] += cumulative_text_length
                if "endIndex" in segment:
                    segment["endIndex"] += cumulative_text_length
            merged_document["entities"].append(entity)

        cumulative_text_length += len(page_text) + 1  # account for newline added

    # Save the merged JSON
    output_path = os.path.join(output_dir, f"{batch_name}.json")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(merged_document, out_f, indent=2)

    print(f"Merged {len(json_files)} pages into {output_path}")
