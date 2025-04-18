import json
from fuzzywuzzy import fuzz
import os

def extract_template_keysets(template_json):
    keysets = []
    for doc in template_json:
        for _, content in doc.items():
            keysets.append(list(content.keys()))
    return keysets

def similarity_score(keys1, keys2):
    scores = []
    for k1 in keys1:
        best = max(fuzz.ratio(k1.lower(), k2.lower()) for k2 in keys2)
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0

def fix_keys_entirely(ocr_data, template_keysets, threshold=70):
    fixed_data = []
    for row in ocr_data:
        current_keys = list(row.keys())
        best_score = 0
        best_template_keys = current_keys

        for template_keys in template_keysets:
            score = similarity_score(current_keys, template_keys)
            if score > best_score:
                best_score = score
                best_template_keys = template_keys

        if best_score >= threshold and len(best_template_keys) == len(current_keys):
            new_row = {new_key: row[old_key] for new_key, old_key in zip(best_template_keys, current_keys)}
        else:
            new_row = row
        fixed_data.append(new_row)
    return fixed_data

def fix_json_keys(json_dir, template_path):
    with open(template_path, 'r', encoding='utf-8') as f:
        template_json = json.load(f)

    template_keysets = extract_template_keysets(template_json)

    for file in os.listdir(json_dir):
        if file.endswith("_output.json"):
            json_path = os.path.join(json_dir, file)
            with open(json_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)

            fixed_data = fix_keys_entirely(ocr_data, template_keysets)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(fixed_data, f, ensure_ascii=False, indent=4)

            print(f"🔧 Đã sửa key cho: {file}")