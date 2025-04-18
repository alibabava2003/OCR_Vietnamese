from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

import cv2
import os
import easyocr
from collections import defaultdict

# ---------------------- EASYOCR DETECT LINE ---------------------- #
def extract_boxes_with_easyocr(image_path, output_folder):
    reader = easyocr.Reader(['vi'], gpu=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    # Resize nếu ảnh nhỏ để OCR dễ nhận hơn
    h, w = image.shape[:2]
    if max(h, w) < 1000:
        image = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_LINEAR)

    result = reader.readtext(image)

    if not result:
        print(f"[WARNING] No text detected in image: {image_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    for idx, (box, text, conf) in enumerate(result):
        if conf < 0.3:
            continue  # bỏ kết quả kém tin cậy

        x_coords = [int(pt[0]) for pt in box]
        y_coords = [int(pt[1]) for pt in box]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        cropped = image[y_min:y_max, x_min:x_max]
        output_path = os.path.join(output_folder, f"{base_filename}_line_{idx}.png")
        cv2.imwrite(output_path, cropped)
        print(f"[{base_filename}][{idx}] Saved: {output_path}")

# ---------------------- LẶP QUA TẤT CẢ BẢNG ---------------------- #
def process_all_tables_detect(input_root_folder, output_root_folder):
    for table_folder in sorted(os.listdir(input_root_folder)):
        table_input_path = os.path.join(input_root_folder, table_folder)
        if os.path.isdir(table_input_path) and table_folder.startswith("table_"):
            print(f"=== Processing {table_folder} ===")
            table_output_path = os.path.join(output_root_folder, table_folder)
            os.makedirs(table_output_path, exist_ok=True)

            image_files = [f for f in os.listdir(table_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in image_files:
                image_path = os.path.join(table_input_path, image_file)
                extract_boxes_with_easyocr(image_path, table_output_path)

# ---------------------- DỰ ĐOÁN TEXT VỚI VIETOCR ---------------------- #
def process_table(table_folder_path, output_txt_path):
    try:
        image_files = [f for f in os.listdir(table_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        cell_dict = defaultdict(list)
        for f in image_files:
            if 'cell_' in f and '_line_' in f:
                cell_id = f.split('_line_')[0]
                cell_dict[cell_id].append(f)

        with open(output_txt_path, 'w', encoding='utf-8') as f_out:
            for cell_id in sorted(cell_dict.keys(), key=lambda x: int(x.split('_')[1])):
                lines = sorted(cell_dict[cell_id], key=lambda x: int(x.split('_line_')[1].split('.')[0]))
                for line_img in lines:
                    img_path = os.path.join(table_folder_path, line_img)
                    img = Image.open(img_path)
                    text = detector.predict(img)
                    f_out.write(f"{line_img}\t{text.strip()}\n")
                    print(f"[{line_img}] => {text.strip()}")

        print(f"✅ Đã lưu: {output_txt_path}")

    except Exception as e:
        print(f"❌ Lỗi với bảng {table_folder_path}: {e}")

# ---------------------- LẶP QUA TẤT CẢ BẢNG ĐỂ VIẾT TXT ---------------------- #
def process_all_tables_recog(input_root, output_root_txt):
    os.makedirs(output_root_txt, exist_ok=True)
    for folder in sorted(os.listdir(input_root)):
        if folder.startswith('table_'):
            table_path = os.path.join(input_root, folder)
            if os.path.isdir(table_path):
                output_txt_path = os.path.join(output_root_txt, f"{folder}.txt")
                print(f"\n🔍 Đang xử lý: {folder}")
                process_table(table_path, output_txt_path)

# ---------------------- KHỞI TẠO VIETOCR ---------------------- #
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = r'transformerocr(5).pth'
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
detector = Predictor(config)

# ---------------------- CHẠY TOÀN BỘ PIPELINE ---------------------- #
input_cells_folder = r"CROP_TABLE\output_cells"
output_lines_folder = r"CROP_TABLE\line_easy"
output_txt_folder = r"CROP_TABLE\output_txt"

process_all_tables_detect(input_cells_folder, output_lines_folder)
process_all_tables_recog(output_lines_folder, output_txt_folder)
