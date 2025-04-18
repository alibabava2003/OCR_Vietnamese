const dropArea = document.querySelector(".drag-area");
const dragText = dropArea.querySelector("header");
const button = dropArea.querySelector("button");
const input = document.querySelector("#file");
const convertBtn = document.querySelector("#convertBtn");

let file; // Lưu trữ file được chọn

// Mở file browser khi click nút
button.onclick = () => input.click();

// Chỉ chấp nhận PDF
input.accept = "application/pdf";

// Khi chọn file
input.addEventListener("change", function () {
    file = this.files[0];
    showFile();
});

dropArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropArea.classList.add("active");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("active");
});

dropArea.addEventListener("drop", (event) => {
    event.preventDefault();
    file = event.dataTransfer.files[0];
    showFile();
});

function showFile() {
    // Fix 1: Kiểm tra cả extension file
    if (file && (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf"))) {
        dragText.textContent = "File has been uploaded successfully";
        convertBtn.disabled = false;
    } else {
        alert("Please only upload PDF file");
        input.value = "";
        file = null;
        dragText.textContent = "Drag & Drop to Upload File";
        convertBtn.disabled = true;
    }
}

// Convert và download
convertBtn.addEventListener("click", () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    fetch("/convert_pdf", {
        method: "POST",
        body: formData,
    })
    .then(res => {
        // Fix 2: Lấy filename từ header server
        const contentDisposition = res.headers.get('Content-Disposition');
        const filename = contentDisposition 
            ? contentDisposition.split('filename=')[1].replace(/"/g, '')
            : file.name.replace(/.pdf$/, '') + '.zip'; // Fallback
        
        return res.blob().then(blob => ({ blob, filename }));
    })
    .then(({ blob, filename }) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename; // Fix 3: Dùng filename từ server
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    })
    .catch(err => {
        console.error(err);
        alert("Lỗi chuyển đổi: " + (err.message || "Vui lòng thử lại sau"));
    });
});