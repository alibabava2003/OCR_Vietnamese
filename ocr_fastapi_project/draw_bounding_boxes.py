from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import cv2
import easyocr
import torch
import re
import fitz 
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from docx import Document
from io import BytesIO
import os
import json
from fastapi import Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import NamedTemporaryFile
import zipfile
import asyncio
import time

# Cấu hình VietOCR
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = r"transformerocr(5).pth"
config['cnn']['pretrained'] = False
config['device'] = 'cuda'
detector = Predictor(config)

print("Torch version:", torch.version)
print("CUDA available:", torch.cuda.is_available())

# Tạo instance EasyOCR Reader
reader = easyocr.Reader(['vi'], gpu=True)

# Hàm xử lý ảnh và nhận diện chữ
def draw_bounding_boxes(image_path, batch_size=30):
    image = cv2.imread(image_path)
    results = reader.readtext(image_path)
    results = sorted(results, key=lambda x: x[0][0][1])
    merged_boxes = []

    for bbox, _,_ in results:
        top_left, top_right, bottom_right, bottom_left = bbox
        x_min, y_min = top_left
        x_max, y_max = bottom_right
        x_max = image.shape[1]
        merged = False

        for merged_box in merged_boxes:
            mx_min, my_min, mx_max, my_max = merged_box
            overlap_height = min(y_max, my_max) - max(y_min, my_min)
            min_height = min(y_max - y_min, my_max - my_min)

            if overlap_height > 0 and (overlap_height / min_height) > 0.865:
                merged_box[0] = min(mx_min, x_min)
                merged_box[1] = min(my_min, y_min)
                merged_box[2] = max(mx_max, x_max)
                merged_box[3] = max(my_max, y_max)
                merged = True
                break

        if not merged:
            merged_boxes.append([x_min, y_min, x_max, y_max])

    extracted_texts = []
    cropped_images = []

    for x_min, y_min, x_max, y_max in merged_boxes:
        cropped_line = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        gray = cv2.cvtColor(cropped_line, cv2.COLOR_BGR2GRAY)
        gray_pil = Image.fromarray(gray)
        cropped_images.append(gray_pil)

        # Vẽ bounding box như cũ
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    # Chia batch 32 dòng/lượt
    for i in range(0, len(cropped_images), 32):
        batch = cropped_images[i:i + batch_size]
        batch_texts = detector.predict_batch(batch)
        extracted_texts.extend([t.strip() for t in batch_texts])

    processed_image_path = "static/processed_images/processed_image.jpg"
    cv2.imwrite(processed_image_path, image)
    return extracted_texts, processed_image_path

def process_outside_table_images(folder_path):
    all_text = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            print(f"Đang nhận diện vùng ngoài bảng: {filename}")
            lines, _ = draw_bounding_boxes(image_path)
            all_text.extend(lines)

    return all_text