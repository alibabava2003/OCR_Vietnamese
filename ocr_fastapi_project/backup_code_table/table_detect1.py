from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import os
import cv2
import numpy as np
import os
import cv2
import json
import numpy as np
from PIL import Image, ImageEnhance

# Load model
processor = DetrImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
model = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")

# Load ảnh
image_path = r"data_ (16).jpg"
image = Image.open(image_path)
image_cv = cv2.imread(image_path)
image_width, image_height = image.size

# Detect bảng
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# Tạo folder lưu ảnh
output_table_dir = r"CROP_TABLE\TABLE_DETECTION"
output_text_dir = r"CROP_TABLE\text_only"
os.makedirs(output_table_dir, exist_ok=True)
os.makedirs(output_text_dir, exist_ok=True)

# Lưu lại bounding boxes bảng sau khi padding
table_boxes = []

# Cắt bảng ra
for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
    box = [round(i, 2) for i in box.tolist()]
    label_name = model.config.id2label[label.item()]
    print(f"Detected {label_name} with confidence {round(score.item(), 3)} at location {box}")

    left, top, right, bottom = box
    left = max(0, left - 25)
    top = max(0, top - 25)
    right = min(image_width, right + 25)
    bottom = min(image_height, bottom + 25)

    # Lưu lại box để cắt phần ngoài bảng
    table_boxes.append((left, top, right, bottom))

    # Cắt bảng
    cropped_image = image.crop((left, top, right, bottom))
    output_path = os.path.join(output_table_dir, f"{label_name}_{idx}.png")
    cropped_image.save(output_path)
    print(f"Saved cropped image to {output_path}")

# Cắt các phần ngoài bảng
def crop_outside_table(image, table_boxes, save_folder):
    width, height = image.size
    PADDING_TOP = 25 # Padding trên bảng
    PADDING_BOTTOM = 20  # Padding dưới bảng

    # Sắp xếp bảng theo tọa độ y
    table_boxes = sorted(table_boxes, key=lambda box: box[1])
    non_table_regions = []

    # Trên bảng đầu tiên
    y1_first = int(table_boxes[0][1])
    if y1_first > 0:
        non_table_regions.append((0, min(height, y1_first + PADDING_TOP)))

    # Giữa các bảng
    for i in range(len(table_boxes) - 1):
        y2_prev = int(table_boxes[i][3])
        y1_next = int(table_boxes[i+1][1])
        y_start = max(0, y2_prev - PADDING_TOP)
        y_end = min(height, y1_next + PADDING_BOTTOM)
        if y_end > y_start:
            non_table_regions.append((y_start, y_end))

    # Dưới bảng cuối cùng
    y2_last = int(table_boxes[-1][3])
    if y2_last < height:
        non_table_regions.append((max(0, y2_last - PADDING_BOTTOM), height))

    # Cắt và lưu
    for idx, (y_start, y_end) in enumerate(non_table_regions):
        crop = image.crop((0, y_start, width, y_end))
        crop.save(os.path.join(save_folder, f"text_{idx}.png"))

    print(f"✅ Đã lưu {len(non_table_regions)} vùng không chứa bảng vào: {save_folder}")

# Gọi hàm cắt phần không chứa bảng
crop_outside_table(image, table_boxes, output_text_dir)

# === Hàm tiền xử lý ===
def preprocess(img, factor: int):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(img).enhance(factor)
    if gray.std() < 30:
        enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
    return np.array(enhancer)

# === Gộp đường ngang ===
def group_h_lines(h_lines, thin_thresh):
    new_h_lines = []
    while len(h_lines) > 0:
        thresh = sorted(h_lines, key=lambda x: x[0][1])[0][0]
        lines = [line for line in h_lines if thresh[1] - thin_thresh <= line[0][1] <= thresh[1] + thin_thresh]
        h_lines = [line for line in h_lines if not (thresh[1] - thin_thresh <= line[0][1] <= thresh[1] + thin_thresh)]
        x = []
        for line in lines:
            x.extend([line[0][0], line[0][2]])
        x_min, x_max = min(x) - int(3 * thin_thresh), max(x) + int(3 * thin_thresh)
        new_h_lines.append([x_min, thresh[1], x_max, thresh[1]])
    return new_h_lines

# === Gộp đường dọc ===
def group_v_lines(v_lines, thin_thresh):
    new_v_lines = []
    while len(v_lines) > 0:
        thresh = sorted(v_lines, key=lambda x: x[0][0])[0][0]
        lines = [line for line in v_lines if thresh[0] - thin_thresh <= line[0][0] <= thresh[0] + thin_thresh]
        v_lines = [line for line in v_lines if not (thresh[0] - thin_thresh <= line[0][0] <= thresh[0] + thin_thresh)]
        y = []
        for line in lines:
            y.extend([line[0][1], line[0][3]])
        y_min, y_max = min(y) - int(3 * thin_thresh), max(y) + int(3 * thin_thresh)
        new_v_lines.append([thresh[0], y_min, thresh[0], y_max])
    return new_v_lines

# === Giao điểm 2 đoạn ===
def seg_intersect(line1, line2):
    a1, a2 = line1
    b1, b2 = line2
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1

    def perp(a): return np.array([-a[1], a[0]])
    dap = perp(da)
    denom = np.dot(dap, db)
    if denom == 0:
        return None, None
    num = np.dot(dap, dp)
    intersect = (num / denom.astype(float)) * db + b1
    return intersect

# === Xác định ô phải dưới ===
def get_bottom_right(right_points, bottom_points, points):
    for right in right_points:
        for bottom in bottom_points:
            if [right[0], bottom[1]] in points:
                return right[0], bottom[1]
    return None, None

# === Thư mục đầu vào và đầu ra ===
input_dir = r"CROP_TABLE\TABLE_DETECTION"
output_base = r"CROP_TABLE\output_cells"

# === Duyệt từng ảnh trong thư mục ===
for file in os.listdir(input_dir):
    if file.lower().endswith(".png"):
        input_image_path = os.path.join(input_dir, file)
        table_name = os.path.splitext(file)[0]
        output_dir_cells = os.path.join(output_base, table_name)
        os.makedirs(output_dir_cells, exist_ok=True)

        table_image = cv2.imread(input_image_path)
        table_image = preprocess(table_image, factor=2)
        gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bin = 255 - img_bin

        kernel_len = gray.shape[1] // 100

        # === Phát hiện và gộp đường ngang ===
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        image_horizontal = cv2.erode(img_bin, hor_kernel, iterations=3)
        horizontal_lines = cv2.dilate(image_horizontal, hor_kernel, iterations=3)
        h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi / 180, 50, maxLineGap=200)
        if h_lines is None:
            h_lines = []
        new_horizontal_lines = group_h_lines(h_lines, kernel_len)

        # === Phát hiện và gộp đường dọc ===
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        image_vertical = cv2.erode(img_bin, ver_kernel, iterations=3)
        vertical_lines = cv2.dilate(image_vertical, ver_kernel, iterations=3)
        v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi / 180, 50, maxLineGap=200)
        if v_lines is None:
            v_lines = []
        new_vertical_lines = group_v_lines(v_lines, kernel_len)

# === Tìm giao điểm để xác định cell ===
        points = []
        for hline in new_horizontal_lines:
            for vline in new_vertical_lines:
                line1 = [np.array(hline[:2]), np.array(hline[2:])]
                line2 = [np.array(vline[:2]), np.array(vline[2:])]
                x, y = seg_intersect(line1, line2)
                if x is not None and hline[0] <= x <= hline[2] and vline[1] <= y <= vline[3]:
                    points.append([int(x), int(y)])

        cells = []
        for point in points:
            left, top = point
            right_points = sorted([p for p in points if p[0] > left and p[1] == top], key=lambda x: x[0])
            bottom_points = sorted([p for p in points if p[1] > top and p[0] == left], key=lambda x: x[1])
            right, bottom = get_bottom_right(right_points, bottom_points, points)
            if right and bottom:
                cv2.rectangle(table_image, (left, top), (right, bottom), (0, 0, 255), 2)
                cells.append([left, top, right, bottom])

        print(f"📄 {file}: Tổng số ô bảng = {len(cells)}")

        # === Lưu cell crop ===
        for idx, cell in enumerate(cells):
            left, top, right, bottom = cell
            cell_img = table_image[top:bottom, left:right]
            cell_path = os.path.join(output_dir_cells, f"cell_{idx}.png")
            cv2.imwrite(cell_path, cell_img)

        # === Lưu danh sách box và số lượng ===
        json_box_path = os.path.join(output_dir_cells, f"{table_name}_boxes.json")
        with open(json_box_path, "w", encoding="utf-8") as f:
            json.dump(cells, f, indent=4, ensure_ascii=False)

        with open(os.path.join(output_dir_cells, "cell_count.txt"), "w", encoding="utf-8") as f:
            f.write(f"{len(cells)}\n")