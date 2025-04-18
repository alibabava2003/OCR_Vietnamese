import json
import os

def load_ocr_lines(txt_path):
    ocr_lines = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                name, text = line.strip().split('\t', 1)
                ocr_lines[name] = text
    return ocr_lines

def get_text(cell_id, ocr_lines):
    lines = []
    for key in ocr_lines:
        if key.startswith(f"cell_{cell_id}_"):
            lines.append(ocr_lines[key])
    return " ".join(lines).strip() if lines else None

def create_json_from_txt(input_json_folder, input_txt_folder, output_json_folder):
    os.makedirs(output_json_folder, exist_ok=True)

    table_num = 0
    while True:
        table_name = f"table_{table_num}"
        json_path = os.path.join(input_json_folder, f"{table_name}/{table_name}_boxes.json")
        txt_path = os.path.join(input_txt_folder, f"{table_name}.txt")
        output_path = os.path.join(output_json_folder, f"{table_name}_output.json")

        if not os.path.exists(json_path) or not os.path.exists(txt_path):
            print(f"❌ Bảng {table_name} không tồn tại. Dừng lại ở bảng này.")
            break

        with open(json_path, "r", encoding="utf-8") as f:
            cell_boxes = json.load(f)

        y2_header = cell_boxes[0][3]
        header_cells = [(i, box) for i, box in enumerate(cell_boxes) if box[3] == y2_header]
        header_cells_sorted = sorted(header_cells, key=lambda x: x[1][0])
        header_cell_indexes = [i for i, _ in header_cells_sorted]
        num_columns = len(header_cell_indexes)

        ocr_lines = load_ocr_lines(txt_path)
        headers = [get_text(i, ocr_lines) or f"Column_{i}" for i in header_cell_indexes]

        data = []
        cell_id = max(header_cell_indexes) + 1
        while cell_id + num_columns - 1 < len(cell_boxes):
            row_cells = cell_boxes[cell_id:cell_id + num_columns]
            row_data = {}
            for col_idx, header in enumerate(headers):
                box = row_cells[col_idx]
                try:
                    full_index = cell_boxes.index(box)
                except ValueError:
                    full_index = None
                text = get_text(full_index, ocr_lines) if full_index is not None else None
                row_data[header] = text
            cell_id += num_columns
            data.append(row_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"✅ Đã tạo file JSON cho {table_name} tại: {output_path}")
        table_num += 1