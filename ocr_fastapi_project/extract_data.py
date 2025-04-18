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

def extract_data(text):
    print("📄 [extract_data] Raw text input (first 200 chars):", text[:200])
    data = {}
    special_patterns = [
        (r"Tôi là \(ghi họ tên bằng chữ in hoa\):\s*([^\s]+(?:\s[^\s]+)*)\s*Giới tính:\s*([^\s]+)", ["Tôi là (ghi họ tên bằng chữ in hoa)", "Giới tính"]),
        (r"Sinh ngày:\s*([\d/]+)\s*Dân tộc:\s*([^\s]+)\s*Quốc tịch:\s*(\S.+)", ["Sinh ngày", "Dân tộc", "Quốc tịch"]),
        (r"Điện thoại \(nếu có\):\s*([\d]*)\s*Fax \(nếu có\):\s*(\d*)", ["Điện thoại (nếu có)", "Fax (nếu có)"]),
        (r"Điện thoại \(nếu có\):\s*([\d]*)\s*Email \(nếu có\):\s*([\w\.-]+@[\w\.-]+)?", ["Điện thoại (nếu có)", "Email (nếu có)"]),
        (r"Email \(nếu có\):\s*([\w\.-]+@[\w\.-]+)?\s*Website \(nếu có\):\s*(\S*)", ["Email (nếu có)", "Website (nếu có)"])
    ]

    # Xử lý các pattern đặc biệt
    for pattern, keys in special_patterns:
        matches = re.findall(pattern, text)
        print(f"Matches for pattern {pattern}: {matches}")  # Debug line
        for match in matches:
            for i, key in enumerate(keys):
                value = match[i].strip() if match[i] else None
                data[key] = value

    # Xử lý các dòng dữ liệu chung
    pattern = r"^(.*?):\s*(.*?)(?=\n|$)"
    matches = re.findall(pattern, text, re.MULTILINE)
    print(f"Matches for general pattern: {matches}")  # Debug line
    for key, value in matches:
        key = key.strip()
        value = value.strip() if value.strip() else None
        if key not in data:  # Tránh ghi đè dữ liệu đã có
            data[key] = value

    # Chuyển đối tượng data thành JSON, đảm bảo không mã hóa ký tự tiếng Việt và định dạng đẹp
    json_data = json.dumps(data, ensure_ascii=False, indent=2)
    print(f"Extracted JSON data: {json_data}")  # Debug line

    return json.loads(json_data)  # Trả về đối tượng JSON