from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
from torchvision import transforms
import cv2
import os
import easyocr
import torch
from collections import defaultdict

# ---------------------- EASY OCR DETECT LINE ---------------------- #
def ocr_line_images(root_input_folder, root_output_folder):
    reader = easyocr.Reader(['vi'], gpu=True)

    for table_folder in sorted(os.listdir(root_input_folder)):
        table_input_path = os.path.join(root_input_folder, table_folder)
        if os.path.isdir(table_input_path) and table_folder.startswith("table_"):
            print(f"=== EasyOCR xử lý {table_folder} ===")
            table_output_path = os.path.join(root_output_folder, table_folder)
            os.makedirs(table_output_path, exist_ok=True)

            image_files = [f for f in os.listdir(table_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in image_files:
                image_path = os.path.join(table_input_path, image_file)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"[ERROR] Không đọc được ảnh: {image_path}")
                    continue

                h, w = image.shape[:2]
                if max(h, w) < 1000:
                    image = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_LINEAR)

                result = reader.readtext(image)
                if not result:
                    print(f"[WARNING] Không phát hiện dòng chữ: {image_path}")
                    continue

                base_filename = os.path.splitext(os.path.basename(image_path))[0]

                for idx, (box, text, conf) in enumerate(result):
                    if conf < 0.3:
                        continue
                    x_coords = [int(pt[0]) for pt in box]
                    y_coords = [int(pt[1]) for pt in box]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    x_min = max(0, min(x_min, image.shape[1]-1))
                    x_max = max(0, min(x_max, image.shape[1]-1))
                    y_min = max(0, min(y_min, image.shape[0]-1))
                    y_max = max(0, min(y_max, image.shape[0]-1))

                    cropped = image[y_min:y_max, x_min:x_max]

                    if cropped is not None and cropped.size > 0:
                        output_path = os.path.join(table_output_path, f"{base_filename}_line_{idx}.png")
                        cv2.imwrite(output_path, cropped)
                        print(f"[{base_filename}][{idx}] Saved: {output_path}")
                    else:
                        print(f"[WARNING] Bỏ qua ảnh rỗng khi crop: {image_path} - box {idx}")

# Hàm resize và normalize ảnh
def imgproc(image, imgH=32, imgMinW=100, imgMaxW=1000):
    aspect_ratio = image.width / image.height
    new_w = int(imgH * aspect_ratio)
    new_w = max(imgMinW, min(new_w, imgMaxW))

    image = image.resize((new_w, imgH), Image.BILINEAR)

    new_image = Image.new('RGB', (imgMaxW, imgH), (255, 255, 255))
    new_image.paste(image, (0, 0))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(new_image)

# ---------------------- VIETOCR TEXT RECOGNITION (BATCH) ---------------------- #
def run_vietocr(input_root, output_root_txt, vietocr_weight_path, batch_size=32):
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = vietocr_weight_path
    config['cnn']['pretrained'] = False
    config['device'] = 'cuda'
    detector = Predictor(config)

    os.makedirs(output_root_txt, exist_ok=True)

    for folder in sorted(os.listdir(input_root)):
        if folder.startswith('table_'):
            table_path = os.path.join(input_root, folder)
            if os.path.isdir(table_path):
                output_txt_path = os.path.join(output_root_txt, f"{folder}.txt")
                print(f"\nVietOCR xử lý theo batch: {folder}")

                image_files = [f for f in os.listdir(table_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'cell_' in f and '_line_' in f]
                image_files.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_line_')[1].split('.')[0])))

                batch_images = []
                batch_filenames = []

                with open(output_txt_path, 'w', encoding='utf-8') as f_out:
                    for i, img_name in enumerate(image_files):
                        img_path = os.path.join(table_path, img_name)
                        img = Image.open(img_path).convert("RGB")
                        batch_images.append(img)
                        batch_filenames.append(img_name)

                        if len(batch_images) == batch_size or i == len(image_files) - 1:
                            # Gọi predict_batch trực tiếp trên list ảnh PIL
                            batch_result = detector.predict_batch(batch_images)

                            for name, text in zip(batch_filenames, batch_result):
                                f_out.write(f"{name}\t{text.strip()}\n")
                                print(f"[{name}] => {text.strip()}")

                            batch_images = []
                            batch_filenames = []

                print(f"Đã lưu: {output_txt_path}")