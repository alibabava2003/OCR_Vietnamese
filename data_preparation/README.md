# Data Preparation for OCR Training

This document outlines the data preparation process for OCR training, detailing collection, preprocessing, labeling, normalization, and dataset splitting.

---

## 📌 1. Data Collection

### 📝 1.1. Document Preparation
- Selected **two common types of administrative documents**.
- Printed **200 copies** (100 per document type).
- Included **various handwritten fields** (names, dates, addresses, etc.).
- Ensured **document diversity** for OCR adaptability.

### 👥 1.2. Participant Involvement
- Recruited **diverse individuals** (students, family, acquaintances).
- Ensured **handwriting style variability**.
- Collected **203 completed documents** with real handwritten data.

### 📊 1.3. Data Overview
- Variety in handwriting styles, ink colors, and writing speeds.
- Dataset provides a **strong foundation for OCR training**.

---

## 🔍 2. Data Preprocessing

### 🖼 2.1. Image Conversion
- **Mobile scanning apps** discarded due to distortions.
- **Professional scanning** used for better quality.
- Applied **noise removal & blurring reduction techniques**.

### ✂️ 2.2. Image Cropping & Bounding Box Methods

| Method         | Accuracy | Issues |
|---------------|---------|--------|
| **EasyOCR**   | ~90%    | Fragmented bounding boxes |
| **PaddleOCR** | ~60-70% | Inconsistent detection |
| **Manual**    | ~100%   | Time-consuming but precise |

- **Total cropped images**: `11,430`
- **Output**: Cropped images + OCR label text files.

---

## 🔠 3. Data Labeling

### ✅ 3.1. OCR Label Review & Correction
- **Manual verification** of all OCR labels.
- Fixed **misinterpreted characters, missing words, punctuation mistakes**.

### ⚡ 3.2. Workflow Optimization
- Structured process for **error reduction & efficiency improvement**.
- Ensured **dataset uniformity** for accurate OCR training.

---

## 🛠 4. Data Normalization

### 📂 4.1. Image Path Normalization
- **Unified image storage** in a single folder.
- **Merged text files** into a structured dataset.

### 🔎 4.2. Error Detection & Correction
- **Inconsistencies found**:
  - `14` missing images.
  - `4` missing text file entries.
  - `102` duplicate images.
- **Final dataset**: `11,416` images with corresponding labels.

### 📏 4.3. Text Format Validation
- Ensured proper format:  
- Removed formatting inconsistencies.

### 📌 4.4. Image Categorization
- **Line-level images**: `3,285`
- **Word-level images**: `8,131`
- Verified **correct label-image pairing**.

---

## 📊 5. Dataset Splitting

### 📌 5.1. Splitting Strategy
- **Training (80%)**, **Validation (10%)**, **Testing (10%)**.
- **Randomized partitioning** to prevent bias.

### 📂 5.2. Data Partitioning
| Image Type | Training | Validation | Testing |
|------------|---------|------------|--------|
| **Line**   | 2,628   | 328        | 328    |
| **Word**   | 6,505   | 813        | 813    |

### 🚀 5.3. Considerations for Future Use
- Structured for **fine-tuning & improvements**.
- Ensures **effective model training, bias prevention, and evaluation reliability**.

---

### 📌 Notes:
- This dataset is **optimized for OCR fine-tuning**.
- All images and labels **strictly follow the preprocessing pipeline**.
- Future enhancements may involve **additional normalization techniques**.

---

🔥 *Maintained by [Your Name/Team]*  
📅 *Last Updated: [Date]*
