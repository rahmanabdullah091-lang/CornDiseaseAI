/**
 * app.js
 * Logika frontend CornDiseaseAI: upload, drag & drop, kamera,
 * pemanggilan API deteksi, rendering hasil, dark mode, toast,
 * dan interaksi UI lainnya. Ditulis dengan JavaScript ES6 murni
 * tanpa framework maupun library CSS eksternal.
 */

(() => {
    "use strict";

    /* ============================================================
       KONSTAN & STATE
       ============================================================ */
    const CLASS_COLORS = {
        "Healthy": "#4CAF50",
        "Common Rust": "#E53935",
        "Gray Leaf Spot": "#1E88E5",
        "Northern Leaf Blight": "#FB8C00",
    };

    const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20 MB
    const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"];

    const state = {
        selectedFile: null,       // File object dari upload
        selectedDataUrl: null,    // Data URL dari hasil capture kamera
        cameraStream: null,
        isDetecting: false,
    };

    /* ============================================================
       DOM ELEMENTS
       ============================================================ */
    const el = {
        loadingScreen: document.getElementById("loadingScreen"),
        toastContainer: document.getElementById("toastContainer"),
        navbar: document.getElementById("navbar"),
        themeToggle: document.getElementById("themeToggle"),
        backToTop: document.getElementById("backToTop"),

        dropZone: document.getElementById("dropZone"),
        fileInput: document.getElementById("fileInput"),
        browseBtn: document.getElementById("browseBtn"),

        cameraViewport: document.getElementById("cameraViewport"),
        cameraStream: document.getElementById("cameraStream"),
        cameraPlaceholder: document.getElementById("cameraPlaceholder"),
        captureCanvas: document.getElementById("captureCanvas"),
        openCameraBtn: document.getElementById("openCameraBtn"),
        captureBtn: document.getElementById("captureBtn"),

        previewCard: document.getElementById("previewCard"),
        previewImage: document.getElementById("previewImage"),
        clearPreviewBtn: document.getElementById("clearPreviewBtn"),
        detectBtn: document.getElementById("detectBtn"),

        resultsWrapper: document.getElementById("resultsWrapper"),
        resultImage: document.getElementById("resultImage"),
        predictionTimeBadge: document.getElementById("predictionTimeBadge"),
        summaryGrid: document.getElementById("summaryGrid"),
        leafCountValue: document.getElementById("leafCountValue"),
        avgConfidenceValue: document.getElementById("avgConfidenceValue"),
        predictionTimeValue: document.getElementById("predictionTimeValue"),
        detailList: document.getElementById("detailList"),
    };

    /* ============================================================
       INIT
       ============================================================ */
    function init() {
        document.getElementById("currentYear").textContent = new Date().getFullYear();

        initTheme();
        initLoadingScreen();
        initNavbarScroll();
        initBackToTop();
        initRippleButtons();
        initUploadHandlers();
        initCameraHandlers();
        initDetectHandlers();

        if (window.__MODEL_READY__ === false) {
            showToast(
                "error",
                "Model belum siap. Pastikan file BEST_MODEL_SIZE_156_SPLIT_20.h5 ada di folder model/."
            );
        }
    }

    /* ============================================================
       LOADING SCREEN
       ============================================================ */
    function initLoadingScreen() {
        window.addEventListener("load", () => {
            setTimeout(() => {
                el.loadingScreen.classList.add("hidden");
            }, 400);
        });
    }

    /* ============================================================
       DARK MODE
       ============================================================ */
    function initTheme() {
        const saved = localStorage.getItem("corn-disease-theme");
        const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        const theme = saved || (prefersDark ? "dark" : "light");
        applyTheme(theme);

        el.themeToggle.addEventListener("click", () => {
            const current = document.body.getAttribute("data-theme");
            const next = current === "dark" ? "light" : "dark";
            applyTheme(next);
            localStorage.setItem("corn-disease-theme", next);
        });
    }

    function applyTheme(theme) {
        document.body.setAttribute("data-theme", theme);
        const icon = el.themeToggle.querySelector("i");
        icon.className = theme === "dark" ? "fa-solid fa-sun" : "fa-solid fa-moon";
    }

    /* ============================================================
       NAVBAR SCROLL EFFECT + BACK TO TOP
       ============================================================ */
    function initNavbarScroll() {
        window.addEventListener("scroll", () => {
            el.navbar.classList.toggle("scrolled", window.scrollY > 12);
        }, { passive: true });
    }

    function initBackToTop() {
        window.addEventListener("scroll", () => {
            el.backToTop.classList.toggle("visible", window.scrollY > 480);
        }, { passive: true });

        el.backToTop.addEventListener("click", () => {
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }

    /* ============================================================
       RIPPLE EFFECT
       ============================================================ */
    function initRippleButtons() {
        document.querySelectorAll(".ripple").forEach((btn) => {
            btn.addEventListener("click", (event) => {
                const rect = btn.getBoundingClientRect();
                const circle = document.createElement("span");
                const size = Math.max(rect.width, rect.height);

                circle.className = "ripple-circle";
                circle.style.width = circle.style.height = `${size}px`;
                circle.style.left = `${event.clientX - rect.left - size / 2}px`;
                circle.style.top = `${event.clientY - rect.top - size / 2}px`;

                btn.appendChild(circle);
                setTimeout(() => circle.remove(), 600);
            });
        });
    }

    /* ============================================================
       TOAST NOTIFICATION
       ============================================================ */
    const TOAST_ICONS = {
        success: "fa-solid fa-circle-check",
        error: "fa-solid fa-circle-exclamation",
        info: "fa-solid fa-circle-info",
    };

    function showToast(type, message, duration = 4200) {
        // Cegah toast yang sama menumpuk berulang-ulang di layar
        const alreadyShown = Array.from(el.toastContainer.children).some(
            (node) => node.dataset.message === message && !node.classList.contains("leaving")
        );
        if (alreadyShown) return;

        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        toast.dataset.message = message;
        toast.innerHTML = `<i class="${TOAST_ICONS[type] || TOAST_ICONS.info}"></i><span>${escapeHtml(message)}</span>`;
        el.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add("leaving");
            setTimeout(() => toast.remove(), 320);
        }, duration);
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    /* ============================================================
       UPLOAD (CLICK + DRAG & DROP)
       ============================================================ */
    function initUploadHandlers() {
        el.browseBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            el.fileInput.click();
        });

        el.dropZone.addEventListener("click", () => el.fileInput.click());

        el.fileInput.addEventListener("change", () => {
            const file = el.fileInput.files[0];
            if (file) handleSelectedFile(file);
        });

        ["dragenter", "dragover"].forEach((eventName) => {
            el.dropZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                event.stopPropagation();
                el.dropZone.classList.add("drag-over");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            el.dropZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                event.stopPropagation();
                el.dropZone.classList.remove("drag-over");
            });
        });

        el.dropZone.addEventListener("drop", (event) => {
            const file = event.dataTransfer.files[0];
            if (file) handleSelectedFile(file);
        });

        el.clearPreviewBtn.addEventListener("click", resetSelection);
    }

    function handleSelectedFile(file) {
        if (!ALLOWED_TYPES.includes(file.type)) {
            showToast("error", "Format file tidak didukung. Gunakan PNG, JPG, JPEG, atau WEBP.");
            return;
        }
        if (file.size > MAX_FILE_SIZE) {
            showToast("error", "Ukuran file melebihi batas maksimal 20 MB.");
            return;
        }

        state.selectedFile = file;
        state.selectedDataUrl = null;
        stopCamera();

        const reader = new FileReader();
        reader.onload = (event) => showPreview(event.target.result);
        reader.readAsDataURL(file);
    }

    /* ============================================================
       CAMERA
       ============================================================ */
    function initCameraHandlers() {
        el.openCameraBtn.addEventListener("click", toggleCamera);
        el.captureBtn.addEventListener("click", captureFromCamera);
    }

    async function toggleCamera() {
        if (state.cameraStream) {
            stopCamera();
            return;
        }

        // Cegah user menekan tombol berkali-kali saat request masih berjalan
        // (mencegah toast error menumpuk seperti yang dilaporkan sebelumnya).
        if (state.isRequestingCamera) return;

        // Deteksi dini: getUserMedia hanya tersedia pada secure context
        // (HTTPS, atau localhost/127.0.0.1). Di origin lain (mis. IP LAN
        // seperti http://192.168.x.x), navigator.mediaDevices akan bernilai
        // undefined dan browser TIDAK akan menampilkan dialog izin sama sekali.
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showToast(
                "error",
                "Kamera tidak tersedia di sini. Buka situs ini lewat http://localhost:5000 atau https://, bukan alamat IP biasa."
            );
            return;
        }

        state.isRequestingCamera = true;
        el.openCameraBtn.disabled = true;

        try {
            let stream;
            try {
                // Preferensi kamera belakang (untuk HP)
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "environment" },
                    audio: false,
                });
            } catch (preferredError) {
                // Fallback: laptop/PC umumnya tidak punya kamera "environment",
                // jadi coba lagi tanpa constraint facingMode.
                if (preferredError.name === "OverconstrainedError" || preferredError.name === "NotFoundError") {
                    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                } else {
                    throw preferredError;
                }
            }

            state.cameraStream = stream;
            el.cameraStream.srcObject = stream;
            el.cameraViewport.classList.add("active");
            el.captureBtn.disabled = false;
            el.openCameraBtn.innerHTML = '<i class="fa-solid fa-video-slash"></i> Tutup Kamera';
        } catch (error) {
            showToast("error", describeCameraError(error));
        } finally {
            state.isRequestingCamera = false;
            el.openCameraBtn.disabled = false;
        }
    }

    function describeCameraError(error) {
        switch (error.name) {
            case "NotAllowedError":
            case "SecurityError":
                return "Izin kamera ditolak. Klik ikon kamera/gembok di address bar browser, ubah menjadi \"Allow\", lalu muat ulang halaman.";
            case "NotFoundError":
            case "DevicesNotFoundError":
                return "Tidak ada kamera yang terdeteksi pada perangkat ini.";
            case "NotReadableError":
            case "TrackStartError":
                return "Kamera sedang dipakai aplikasi lain (mis. Zoom, aplikasi kamera bawaan). Tutup aplikasi tersebut lalu coba lagi.";
            case "OverconstrainedError":
                return "Tidak ada kamera yang cocok dengan pengaturan yang diminta.";
            default:
                return `Tidak dapat mengakses kamera (${error.name || "Unknown"}). Periksa izin kamera pada browser Anda.`;
        }
    }

    function stopCamera() {
        if (state.cameraStream) {
            state.cameraStream.getTracks().forEach((track) => track.stop());
            state.cameraStream = null;
        }
        el.cameraStream.srcObject = null;
        el.cameraViewport.classList.remove("active");
        el.captureBtn.disabled = true;
        el.openCameraBtn.innerHTML = '<i class="fa-solid fa-video"></i> Buka Kamera';
    }

    function captureFromCamera() {
        const video = el.cameraStream;
        const canvas = el.captureCanvas;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const dataUrl = canvas.toDataURL("image/jpeg", 0.92);

        state.selectedDataUrl = dataUrl;
        state.selectedFile = null;
        stopCamera();

        showPreview(dataUrl);
        showToast("success", "Foto berhasil diambil dari kamera.");
    }

    /* ============================================================
       PREVIEW
       ============================================================ */
    function showPreview(dataUrl) {
        el.previewImage.src = dataUrl;
        el.previewCard.hidden = false;
        el.resultsWrapper.hidden = true;
        el.previewCard.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    function resetSelection() {
        state.selectedFile = null;
        state.selectedDataUrl = null;
        el.fileInput.value = "";
        el.previewCard.hidden = true;
        el.resultsWrapper.hidden = true;
    }

    /* ============================================================
       DETEKSI (PANGGIL API BACKEND)
       ============================================================ */
    function initDetectHandlers() {
        el.detectBtn.addEventListener("click", runDetection);
    }

    async function runDetection() {
        if (state.isDetecting) return;
        if (!state.selectedFile && !state.selectedDataUrl) {
            showToast("error", "Silakan pilih atau ambil gambar terlebih dahulu.");
            return;
        }

        state.isDetecting = true;
        setDetectingUI(true);

        try {
            const response = await fetch("/api/detect", buildRequestOptions());
            const data = await response.json();

            if (!response.ok || !data.success) {
                showToast("error", data.message || "Deteksi gagal. Silakan coba lagi.");
                return;
            }

            renderResults(data);
            showToast("success", `Deteksi selesai. ${data.leaf_count} daun terdeteksi.`);
        } catch (error) {
            showToast("error", "Tidak dapat terhubung ke server. Periksa koneksi Anda.");
        } finally {
            state.isDetecting = false;
            setDetectingUI(false);
        }
    }

    function buildRequestOptions() {
        if (state.selectedFile) {
            const formData = new FormData();
            formData.append("image", state.selectedFile);
            return { method: "POST", body: formData };
        }

        return {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_base64: state.selectedDataUrl }),
        };
    }

    function setDetectingUI(isLoading) {
        el.detectBtn.disabled = isLoading;
        el.detectBtn.innerHTML = isLoading
            ? '<i class="fa-solid fa-spinner fa-spin"></i> Menganalisis Gambar...'
            : '<i class="fa-solid fa-brain"></i> Deteksi Penyakit Sekarang';
    }

    /* ============================================================
       RENDER HASIL
       ============================================================ */
    function renderResults(data) {
        el.resultImage.src = data.result_image_url;
        el.predictionTimeBadge.textContent = `${data.prediction_time_ms} ms`;
        el.leafCountValue.textContent = data.leaf_count;
        el.avgConfidenceValue.textContent = `${data.average_confidence}%`;
        el.predictionTimeValue.textContent = `${data.prediction_time_ms} ms`;

        renderSummaryGrid(data.summary);
        renderDetailList(data.detections, data.disease_info);

        el.resultsWrapper.hidden = false;
        el.resultsWrapper.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderSummaryGrid(summary) {
        el.summaryGrid.innerHTML = "";
        Object.entries(summary).forEach(([label, count]) => {
            const chip = document.createElement("div");
            chip.className = "summary-chip";
            chip.innerHTML = `
                <span class="dot" style="background:${CLASS_COLORS[label]}"></span>
                <span>
                    <span class="chip-label">${label}</span>
                    <span class="chip-value">${count}</span>
                </span>
            `;
            el.summaryGrid.appendChild(chip);
        });
    }

    function renderDetailList(detections, diseaseInfo) {
        el.detailList.innerHTML = "";

        if (detections.length === 0) {
            el.detailList.innerHTML = `<p>Tidak ada daun yang berhasil dianalisis pada gambar ini.</p>`;
            return;
        }

        detections.forEach((detection, index) => {
            const info = diseaseInfo[detection.label];
            const item = document.createElement("div");
            item.className = "detail-item";

            item.innerHTML = `
                <div class="detail-item-head" data-toggle>
                    <div class="detail-item-title">
                        <span class="dot" style="background:${detection.color}"></span>
                        <span>Daun #${index + 1} — ${detection.label}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:14px;">
                        <span class="detail-item-conf">${detection.confidence}%</span>
                        <i class="fa-solid fa-chevron-down chevron"></i>
                    </div>
                </div>
                <div class="detail-item-body">
                    <span class="severity-tag severity-${info.severity_level}">Keparahan: ${info.severity}</span>
                    <div class="detail-row">
                        <span class="row-label">Deskripsi</span>
                        <span class="row-value">${info.description}</span>
                    </div>
                    <div class="detail-row">
                        <span class="row-label">Gejala</span>
                        <span class="row-value">${info.symptoms}</span>
                    </div>
                    <div class="detail-row">
                        <span class="row-label">Penyebab</span>
                        <span class="row-value">${info.causes}</span>
                    </div>
                    <div class="detail-row">
                        <span class="row-label">Cara Penanganan</span>
                        <span class="row-value">${info.treatment}</span>
                    </div>
                    <div class="detail-row">
                        <span class="row-label">Rekomendasi Fungisida</span>
                        <span class="row-value">${info.fungicide_recommendation}</span>
                    </div>
                    <div class="detail-row">
                        <span class="row-label">Distribusi Probabilitas</span>
                        <div class="prob-bars">${buildProbabilityBars(detection.probabilities)}</div>
                    </div>
                </div>
            `;

            item.querySelector("[data-toggle]").addEventListener("click", () => {
                item.classList.toggle("open");
            });

            el.detailList.appendChild(item);
        });

        // Buka item pertama secara default
        const firstItem = el.detailList.querySelector(".detail-item");
        if (firstItem) firstItem.classList.add("open");
    }

    function buildProbabilityBars(probabilities) {
        return Object.entries(probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([label, value]) => `
                <div class="prob-bar-row">
                    <span>${label}</span>
                    <span class="prob-bar-track">
                        <span class="prob-bar-fill" style="width:${value}%; background:${CLASS_COLORS[label]}"></span>
                    </span>
                    <span>${value}%</span>
                </div>
            `)
            .join("");
    }

    /* ============================================================
       START
       ============================================================ */
    document.addEventListener("DOMContentLoaded", init);
})();
