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

# Hàm lưu file JSON
def save_json(data, image_name):
    filename = f"{image_name}.json"
    file_path = os.path.join("static/output_json", filename)

    # Debug: kiểm tra file path
    print(f"Saving file to: {file_path}")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"file_path": file_path}


# Hàm lưu file Word
def save_word(text):
    file_path = "static/output_docx/output.docx"  # Lưu file vào thư mục static
    doc = Document()
    doc.add_paragraph(text)
    doc.save(file_path)
    return {"file_path": file_path}