document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const previewImage = document.getElementById("previewImage");
    const noImageText = document.getElementById("noImageText");
    const uploadForm = document.getElementById("uploadForm");
    const spinner = document.getElementById("spinner");
    const imageBox = document.getElementById("imageBox");
    const ocrBox = document.getElementById("ocrBox");

    let currentImageName = null; // ✅ Biến toàn cục để lưu tên ảnh
    let isTableMode = false;

    // ==== 1. Preview ảnh ====
    fileInput.addEventListener("change", function (event) {
        const file = event.target.files[0];
        if (file) {
            imageBox.innerHTML = '';
            ocrBox.innerHTML = '';
            currentImageName = file.name.split(".")[0]; // ✅ Lưu tên ảnh

            const reader = new FileReader();
            reader.onload = function (e) {
                previewImage.src = e.target.result;
                previewImage.style.display = "block";
                if (noImageText) noImageText.style.display = "none";
                imageBox.appendChild(previewImage);
            };
            reader.readAsDataURL(file);
        }
    });

    // ==== 2. Xử lý OCR Text Only ====
    uploadForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        isTableMode = false;
        const file = fileInput.files[0];
        if (!file) return;

        currentImageName = file.name.split(".")[0]; // ✅ Lưu tên ảnh

        spinner.style.display = "block";
        ocrBox.innerHTML = '';

        try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch("/stream_process/", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error(await response.text());

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        appendOCRLine(data);
                    } catch (e) {
                        console.error('Lỗi parse JSON:', e, line);
                    }
                }
            }
        } catch (error) {
            alert("❌ Lỗi: " + error.message);
            console.error('Stream error:', error);
        } finally {
            spinner.style.display = "none";
        }
    });

    // ==== 3. OCR Table (stream bảng + text) ====
    document.getElementById("withTableButton").addEventListener("click", async function (e) {
        e.preventDefault();
        isTableMode = true;
        const file = fileInput.files[0];
        if (!file) {
            alert("Vui lòng chọn ảnh!");
            return;
        }

        currentImageName = file.name.split(".")[0]; // ✅ Lưu tên ảnh

        spinner.style.display = "block";
        ocrBox.innerHTML = '';

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/stream_process_table/", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error(await response.text());

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const parts = buffer.split('\n');
                buffer = parts.pop();

                for (const line of parts) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        if (data.type === "table") {
                            renderTable(data.data, data.table_id);
                        } else {
                            appendOCRLine(data);
                        }
                    } catch (e) {
                        console.error("Lỗi parse JSON:", e, line);
                    }
                }
            }
        } catch (error) {
            alert("❌ Lỗi: " + error.message);
            console.error(error);
        } finally {
            spinner.style.display = "none";
        }
    });

    // ==== 4. Export JSON ====
    document.getElementById("exportJson").addEventListener("click", async function () {
        if (!currentImageName) {
            alert("Vui lòng tải lên một hình ảnh trước.");
            return;
        }

        const exportData = [];
        const elements = ocrBox.childNodes;

        elements.forEach(el => {
            if (el.classList.contains("stream-line")) {
                const text = el.innerText.trim();
                if (text) {
                    exportData.push({ type: "text", text: text });
                }
            }

            if (el.tagName === "DIV" && el.querySelector("table.ocr-table")) {
                const table = el.querySelector("table.ocr-table");
                const headers = Array.from(table.querySelectorAll("th")).map(th => th.textContent.trim());
                const rows = [];

                table.querySelectorAll("tr").forEach((tr, idx) => {
                    if (idx === 0) return;
                    const cells = Array.from(tr.querySelectorAll("td")).map(td => td.textContent.trim());
                    const rowObj = {};
                    headers.forEach((header, i) => {
                        rowObj[header] = cells[i] || "";
                    });
                    rows.push(rowObj);
                });

                exportData.push({ type: "table", data: rows });
            }
        });

        // debug
        console.log("➡️ isTableMode:", isTableMode);
        console.log("➡️ exportData gửi lên:", exportData);

        const response = await fetch("/export_json/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(
                isTableMode 
                ? { data: exportData, image_name: currentImageName } 
                : { text: ocrBox.innerText.trim(), image_name: currentImageName }
            )
        });
        

        if (!response.ok) {
            alert("Xuất file JSON thất bại");
            return;
        }

        const data = await response.json();
        const blob = await fetch(data.file_path).then(res => res.blob());

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = data.file_name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    // ==== 5. Export DOCX ====
    document.getElementById("exportDocx").addEventListener("click", async function () {
        if (!currentImageName) {
            alert("Vui lòng tải lên một hình ảnh trước.");
            return;
        }

        const exportData = [];
        const elements = ocrBox.childNodes;

        elements.forEach(el => {
            if (el.classList.contains("stream-line")) {
                const text = el.innerText.trim();
                if (text) {
                    exportData.push({ type: "text", text: text });
                }
            }

            if (el.tagName === "DIV" && el.querySelector("table.ocr-table")) {
                const table = el.querySelector("table.ocr-table");
                const headers = Array.from(table.querySelectorAll("th")).map(th => th.textContent.trim());
                const rows = [];

                table.querySelectorAll("tr").forEach((tr, idx) => {
                    if (idx === 0) return;
                    const cells = Array.from(tr.querySelectorAll("td")).map(td => td.textContent.trim());
                    const rowObj = {};
                    headers.forEach((header, i) => {
                        rowObj[header] = cells[i] || "";
                    });
                    rows.push(rowObj);
                });

                exportData.push({ type: "table", data: rows });
            }
        });

        const response = await fetch("/export_docx/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ data: exportData, image_name: currentImageName })
        });

        if (!response.ok) {
            alert("Xuất file DOCX thất bại");
            return;
        }

        const data = await response.json();
        const blob = await fetch(data.file_path).then(res => res.blob());

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = data.file_name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    // ==== 6. Hiển thị dòng văn bản ====
    function appendOCRLine(data) {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'stream-line';
        lineDiv.innerHTML = `<span class="line-text">${data.text}</span>`;
        ocrBox.appendChild(lineDiv);
        lineDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // ==== 7. Hiển thị bảng HTML ====
    function renderTable(rows) {
        if (!rows.length) return;

        const wrapper = document.createElement("div");
        wrapper.className = "ocr-table-wrapper";

        const table = document.createElement("table");
        table.className = "ocr-table";

        const headers = Object.keys(rows[0]);
        const headerRow = document.createElement("tr");
        headers.forEach(key => {
            const th = document.createElement("th");
            th.textContent = key;
            headerRow.appendChild(th);
        });
        table.appendChild(headerRow);

        rows.forEach(row => {
            const tr = document.createElement("tr");
            headers.forEach(key => {
                const td = document.createElement("td");
                td.textContent = row[key] ?? "";
                tr.appendChild(td);
            });
            table.appendChild(tr);
        });

        wrapper.appendChild(table);
        ocrBox.appendChild(wrapper);
        wrapper.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // ==== 8. Copy Text ====
    document.getElementById("copyTextBtn").addEventListener("click", () => {
        const text = document.getElementById("ocrBox").innerText.trim();
        if (text) {
            navigator.clipboard.writeText(text).then(() => {
                const toast = document.getElementById("copyToast");
                toast.classList.add("show");

                const ocrBox = document.getElementById("ocrBox");
                ocrBox.classList.add("flash");
                setTimeout(() => ocrBox.classList.remove("flash"), 600);
                setTimeout(() => toast.classList.remove("show"), 1500);
            });
        }
    });
});
