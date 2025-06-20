import os
import json

input_dir = "/merged_outputs"
output_dir = "/simplified_outputs"
os.makedirs(output_dir, exist_ok=True)

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".json"):
        continue

    input_path = os.path.join(input_dir, fname)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract only plain text (discard pages and layout data)
    simplified_data = {
        "batch_name": data.get("batch_name", fname.replace('.json', '')),
        "text": data.get("text", "")
    }

    output_path = os.path.join(output_dir, fname)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_data, f, ensure_ascii=False, indent=2)

    print(f"Simplified and saved: {output_path}")
