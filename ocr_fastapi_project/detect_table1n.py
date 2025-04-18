from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageEnhance
import os
import json
import cv2
import numpy as np

# Phiên bản cập nhật detect_and_crop_tables với return thêm order list
def detect_and_crop_tables(image_path, output_dir):
    processor = DetrImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
    model = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Lưu bounding box vào file JSON
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(output_dir, f"{image_name}_boxes.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([box.tolist() for box in results["boxes"]], f, indent=4, ensure_ascii=False)

    os.makedirs(output_dir, exist_ok=True)
    table_paths = []
    order_list = []

    table_info = []
    for idx, box in enumerate(results["boxes"]):
        y_top = float(box[1])  # lấy toạ độ top
        table_info.append((idx, y_top, box))

    table_info = sorted(table_info, key=lambda x: x[1])

    for idx, _, box in table_info:
        left, top, right, bottom = [round(i, 2) for i in box.tolist()]
        left = max(0, left - 30)
        top = max(0, top - 30)
        right = min(image.width, right + 30)
        bottom = min(image.height, bottom + 30)
        cropped_image = image.crop((left, top, right, bottom))

        table_path = os.path.join(output_dir, f"table_{idx}.png")
        cropped_image.save(table_path)
        print(f"Saved cropped image to {table_path}")
        table_paths.append(table_path)

        order_list.append({
        "type": "table",
        "id": f"table_{idx}",
        "order": top  # 👈 top là y toạ độ trên ảnh, đã có sẵn
    })

    return table_paths, order_list


def detect_and_crop_cells(table_paths, base_output_dir):
    def preprocess(img, factor: int):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Sharpness(img).enhance(factor)
        if gray.std() < 30:
            enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
        return np.array(enhancer)

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

    def get_bottom_right(right_points, bottom_points, points):
        for right in right_points:
            for bottom in bottom_points:
                if [right[0], bottom[1]] in points:
                    return right[0], bottom[1]
        return None, None

    input_dir = os.path.dirname(table_paths[0])
    output_base = os.path.join(base_output_dir, "output_cells")
    os.makedirs(output_base, exist_ok=True)

    for file in table_paths:
        if file.lower().endswith(".png"):
            input_image_path = file
            table_name = os.path.splitext(os.path.basename(file))[0]
            output_dir_cells = os.path.join(output_base, table_name)
            os.makedirs(output_dir_cells, exist_ok=True)

            table_image = cv2.imread(input_image_path)
            table_image = preprocess(table_image, factor=2)
            gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
            _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img_bin = 255 - img_bin

            kernel_len = gray.shape[1] // 100

            hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
            image_horizontal = cv2.erode(img_bin, hor_kernel, iterations=3)
            horizontal_lines = cv2.dilate(image_horizontal, hor_kernel, iterations=3)
            h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi / 180, 50, maxLineGap=200)
            if h_lines is None:
                h_lines = []
            new_horizontal_lines = group_h_lines(h_lines, kernel_len)

            ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
            image_vertical = cv2.erode(img_bin, ver_kernel, iterations=3)
            vertical_lines = cv2.dilate(image_vertical, ver_kernel, iterations=3)
            v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi / 180, 50, maxLineGap=200)
            if v_lines is None:
                v_lines = []
            new_vertical_lines = group_v_lines(v_lines, kernel_len)

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
                    cell_img = table_image[top:bottom, left:right]
                    cell_path = os.path.join(output_dir_cells, f"cell_{len(cells)-1}.png")
                    cv2.imwrite(cell_path, cell_img)

            json_box_path = os.path.join(output_dir_cells, f"{table_name}_boxes.json")
            with open(json_box_path, "w", encoding="utf-8") as f:
                json.dump(cells, f, indent=4, ensure_ascii=False)

            with open(os.path.join(output_dir_cells, "cell_count.txt"), "w", encoding="utf-8") as f:
                f.write(f"{len(cells)}\n")

            print(f"📄 {table_name}.png: Tổng số ô bảng = {len(cells)}")

    return {
        "cell_dir": output_base,
        "line_dir": os.path.join(base_output_dir, "line_paddle"),
        "txt_dir": os.path.join(base_output_dir, "output_txt"),
        "json_dir": os.path.join(base_output_dir, "result_json")
    }

def crop_outside_table(image, table_boxes, save_folder):
    width, height = image.size
    PADDING_TOP = 17
    PADDING_BOTTOM = 20
    table_boxes = sorted(table_boxes, key=lambda box: box[1])
    non_table_regions = []

    y1_first = int(table_boxes[0][1])
    if y1_first > 0:
        non_table_regions.append((0, min(height, y1_first + PADDING_TOP)))

    for i in range(len(table_boxes) - 1):
        y2_prev = int(table_boxes[i][3])
        y1_next = int(table_boxes[i + 1][1])
        y_start = max(0, y2_prev - PADDING_TOP)
        y_end = min(height, y1_next + PADDING_BOTTOM)
        if y_end > y_start:
            non_table_regions.append((y_start, y_end))

    y2_last = int(table_boxes[-1][3])
    if y2_last < height:
        non_table_regions.append((max(0, y2_last - PADDING_BOTTOM), height))

    results = []
    for idx, (y_start, y_end) in enumerate(non_table_regions):
        crop = image.crop((0, y_start, width, y_end))
        path = os.path.join(save_folder, f"text_{idx}.png")
        crop.save(path)
        results.append({
            "path": path,
            "y": y_start
        })

    print(f"✅ Đã lưu {len(non_table_regions)} vùng không chứa bảng vào: {save_folder}")
    return results
