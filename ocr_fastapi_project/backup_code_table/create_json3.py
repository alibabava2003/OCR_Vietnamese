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

# === Đường dẫn gốc
input_json_folder = r"CROP_TABLE/output_cells/"
input_txt_folder = r"CROP_TABLE/output_txt/"
output_json_folder = r"CROP_TABLE/result_json/"

# === Lặp qua tất cả các bảng (table_0, table_1, ...)
table_num = 0
while True:
    table_name = f"table_{table_num}"

    # Đường dẫn tới file JSON và TXT cho bảng hiện tại
    json_path = os.path.join(input_json_folder, f"{table_name}/{table_name}_boxes.json")
    txt_path = os.path.join(input_txt_folder, f"{table_name}.txt")
    output_path = os.path.join(output_json_folder, f"{table_name}_output.json")

    # Kiểm tra xem các file có tồn tại không
    if not os.path.exists(json_path) or not os.path.exists(txt_path):
        print(f"❌ Bảng {table_name} không tồn tại. Dừng lại ở bảng này.")
        break  # Dừng vòng lặp nếu không tìm thấy bảng tiếp theo

    # === Load danh sách bbox từ file JSON
    with open(json_path, "r", encoding="utf-8") as f:
        cell_boxes = json.load(f)

    # === Xác định các cell thuộc dòng tiêu đề: cùng y2 với cell đầu
    y2_header = cell_boxes[0][3]
    header_cells = [(i, box) for i, box in enumerate(cell_boxes) if box[3] == y2_header]

    # Sắp xếp theo x1 để đúng thứ tự từ trái sang phải
    header_cells_sorted = sorted(header_cells, key=lambda x: x[1][0])
    header_cell_indexes = [i for i, _ in header_cells_sorted]
    num_columns = len(header_cell_indexes)

    # === Load text từ file TXT
    ocr_lines = load_ocr_lines(txt_path)

    # === Tên cột
    headers = [get_text(i, ocr_lines) or f"Column_{i}" for i in header_cell_indexes]

    # === Ghi dữ liệu từng dòng
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

        data.append(row_data)
        cell_id += num_columns

    # === Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # === Xuất JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Đã tạo file JSON cho {table_name} tại: {output_path}")
    
    table_num += 1  # Tiếp tục xử lý bảng tiếp theo