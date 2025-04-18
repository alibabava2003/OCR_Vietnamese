import os
import json
from typing import Optional
from PIL import Image

from detect_table1n import detect_and_crop_tables, detect_and_crop_cells, crop_outside_table
from recog_table2n import ocr_line_images, run_vietocr
from create_json_table3n import create_json_from_txt
from correct_json_table4n import fix_json_keys

# Ẩn các cảnh báo trong terminal cho đỡ rối mắt
import warnings
warnings.filterwarnings("ignore")


def full_pipeline(
    image_path: str,
    output_dir: str,
    vietocr_weight_path: str,
    template_path: Optional[str] = None
) -> str:
    print("🚀 Bắt đầu pipeline OCR...")

    # === Bước 1: Phát hiện bảng và cắt ảnh bảng ===
    table_paths, order_list = detect_and_crop_tables(image_path, os.path.join(output_dir, "tables"))
    print(f"📦 Phát hiện {len(table_paths)} bảng")

    # === Bước 2: Cắt vùng ngoài bảng ===
    image = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    box_json_path = os.path.join(output_dir, "tables", f"{image_name}_boxes.json")
    if os.path.exists(box_json_path):
        with open(box_json_path, "r", encoding="utf-8") as f:
            table_boxes = json.load(f)

        outside_folder = os.path.join(output_dir, "outside_table")
        os.makedirs(outside_folder, exist_ok=True)
        text_regions = crop_outside_table(image, table_boxes, outside_folder)

        # Ghi vào order_list từng text với y_start thật
        for info in text_regions:
            order_list.append({
                "type": "text",
                "path": info["path"],
                "order": info["y"]
            })

    # === Ghi order_list theo thứ tự y-top tăng dần (text và bảng) ===
    order_list_sorted = sorted(order_list, key=lambda x: x.get("order", 0))
    with open(os.path.join(output_dir, "order_list.json"), "w", encoding="utf-8") as f:
        json.dump(order_list_sorted, f, indent=2, ensure_ascii=False)

    # === Bước 3: Cắt cell từ bảng ===
    cell_info = detect_and_crop_cells(table_paths, output_dir)

    # === Bước 4: OCR từng dòng trong cell (PaddleOCR + VietOCR) ===
    ocr_line_images(cell_info['cell_dir'], cell_info['line_dir'])
    run_vietocr(cell_info['line_dir'], cell_info['txt_dir'], vietocr_weight_path)

    # === Bước 5: Xoá file *_output.json cũ trước khi tạo lại ===
    json_output_folder = cell_info['json_dir']
    if os.path.exists(json_output_folder):
        for fname in os.listdir(json_output_folder):
            if fname.endswith("_output.json"):
                os.remove(os.path.join(json_output_folder, fname))
                print(f"🧹 Đã xoá file json lẻ cũ: {fname}")

    # === Bước 6: Tạo lại file *_output.json từng bảng ===
    create_json_from_txt(
        input_json_folder=cell_info['cell_dir'],
        input_txt_folder=cell_info['txt_dir'],
        output_json_folder=cell_info['json_dir']
    )

    # === Bước 7: Fix key (nếu có template) ===
    if template_path:
        fix_json_keys(cell_info['json_dir'], template_path)

    # === Bước 8: Gộp toàn bộ JSON thành 1 file duy nhất ===
    combined_data = []
    for filename in sorted(os.listdir(cell_info['json_dir'])):
        if filename.endswith("_output.json"):
            file_path = os.path.join(cell_info['json_dir'], filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                combined_data.extend(data)

    final_json_path = os.path.join(output_dir, "result_json", f"{image_name}_table_output.json")
    os.makedirs(os.path.dirname(final_json_path), exist_ok=True)
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Đã tạo file JSON gộp: {final_json_path}")
    return final_json_path