"""
app.py

Entry point aplikasi Flask untuk CornDiseaseAI.

Alur request '/api/detect':
    1. Terima gambar (file upload ATAU base64 dari kamera)
    2. Decode menjadi array OpenCV (BGR)
    3. detector.py  -> segmentasi & temukan seluruh kandidat daun
    4. predictor.py -> klasifikasikan setiap daun menggunakan Custom CNN
    5. detector.py  -> gambar bounding box + label + confidence
    6. Simpan gambar hasil ke static/results/
    7. Kembalikan JSON berisi ringkasan hasil ke frontend

Jalankan dengan:
    python app.py
"""

from __future__ import annotations

import base64
import os
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from utils import CLASS_COLORS_BGR, CLASS_COLORS_HEX, CLASS_NAMES
from utils.detector import detect_leaves, draw_bounding_box
from utils.predictor import CornDiseasePredictor, ModelNotLoadedError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Model dimuat satu kali saat aplikasi start (singleton), bukan setiap request.
predictor = CornDiseasePredictor.get_instance()

# --------------------------------------------------------------------------
# Informasi detail penyakit — ditampilkan pada bagian "Prediction Detail"
# --------------------------------------------------------------------------
DISEASE_INFO = {
    "Healthy": {
        "description": (
            "Daun jagung dalam kondisi sehat, tidak menunjukkan gejala "
            "infeksi jamur maupun bakteri."
        ),
        "symptoms": "Warna daun hijau merata, tidak ada bercak, lesi, atau perubahan tekstur.",
        "causes": "Tidak ada patogen yang terdeteksi pada permukaan daun.",
        "treatment": (
            "Tidak diperlukan penanganan khusus. Lanjutkan praktik budidaya "
            "yang baik seperti rotasi tanaman dan pemupukan seimbang."
        ),
        "fungicide_recommendation": "Tidak diperlukan aplikasi fungisida.",
        "severity": "Tidak Ada",
        "severity_level": 0,
    },
    "Common Rust": {
        "description": (
            "Penyakit karat daun yang disebabkan oleh jamur Puccinia sorghi, "
            "ditandai dengan bintik-bintik berwarna coklat kemerahan."
        ),
        "symptoms": (
            "Muncul pustula (bintik) kecil berwarna coklat kemerahan hingga "
            "coklat tua pada kedua permukaan daun, tersebar merata."
        ),
        "causes": "Infeksi jamur Puccinia sorghi, berkembang pesat pada kondisi lembab dan suhu sejuk (16-23°C).",
        "treatment": (
            "Gunakan varietas tahan karat, jaga jarak tanam agar sirkulasi "
            "udara baik, dan segera aplikasikan fungisida bila gejala meluas."
        ),
        "fungicide_recommendation": "Fungisida berbahan aktif Azoxystrobin atau Propiconazole.",
        "severity": "Sedang",
        "severity_level": 2,
    },
    "Gray Leaf Spot": {
        "description": (
            "Penyakit bercak daun abu-abu yang disebabkan oleh jamur "
            "Cercospora zeae-maydis, umum terjadi pada iklim lembab."
        ),
        "symptoms": (
            "Bercak persegi panjang berwarna abu-abu hingga coklat sejajar "
            "dengan tulang daun, dapat menyatu dan mengeringkan seluruh daun."
        ),
        "causes": "Infeksi jamur Cercospora zeae-maydis, dipicu kelembaban tinggi dan sisa tanaman terinfeksi di lahan.",
        "treatment": (
            "Rotasi tanaman, olah tanah untuk mengurai sisa tanaman "
            "terinfeksi, dan tanam varietas dengan ketahanan genetik."
        ),
        "fungicide_recommendation": "Fungisida berbahan aktif Pyraclostrobin atau Azoxystrobin + Propiconazole.",
        "severity": "Tinggi",
        "severity_level": 3,
    },
    "Northern Leaf Blight": {
        "description": (
            "Penyakit hawar daun utara yang disebabkan oleh jamur "
            "Exserohilum turcicum, dapat menyebabkan kehilangan hasil signifikan."
        ),
        "symptoms": (
            "Lesi berbentuk cerutu (elips memanjang) berwarna abu-abu "
            "kehijauan hingga coklat, biasanya dimulai dari daun bagian bawah."
        ),
        "causes": "Infeksi jamur Exserohilum turcicum, berkembang pada suhu sedang (18-27°C) dengan kelembaban tinggi.",
        "treatment": (
            "Gunakan benih bersertifikat tahan penyakit, terapkan rotasi "
            "tanaman, dan kelola sisa tanaman pasca panen dengan baik."
        ),
        "fungicide_recommendation": "Fungisida berbahan aktif Mancozeb atau kombinasi Triazole + Strobilurin.",
        "severity": "Tinggi",
        "severity_level": 3,
    },
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_upload_file(file_storage) -> np.ndarray:
    """Decode file upload (multipart/form-data) menjadi array OpenCV BGR."""
    file_bytes = np.frombuffer(file_storage.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Gambar tidak dapat dibaca. Pastikan format file valid.")
    return image


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode gambar hasil capture kamera (data URL base64) menjadi array OpenCV BGR."""
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    binary_data = base64.b64decode(base64_string)
    file_bytes = np.frombuffer(binary_data, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Gambar dari kamera tidak dapat diproses.")
    return image


def run_detection_pipeline(image_bgr: np.ndarray) -> dict:
    """Menjalankan seluruh pipeline: segmentasi -> klasifikasi -> bounding box.

    Returns:
        dict berisi gambar hasil anotasi + daftar deteksi + ringkasan.
    """
    start_time = time.perf_counter()

    annotated = image_bgr.copy()
    candidates = detect_leaves(image_bgr)

    # Fallback: jika OpenCV tidak menemukan kontur daun yang valid
    # (misalnya foto close-up satu daun penuh), gunakan seluruh gambar
    # sebagai satu kandidat agar sistem tetap memberikan hasil.
    if not candidates:
        height, width = image_bgr.shape[:2]
        from utils.detector import LeafCandidate

        candidates = [
            LeafCandidate(x=0, y=0, w=width, h=height, crop_bgr=image_bgr, contour_area=float(width * height))
        ]

    detections = []
    confidence_sum = 0.0

    for candidate in candidates:
        result = predictor.predict(candidate.crop_bgr)

        color_bgr = CLASS_COLORS_BGR[result.label]
        draw_bounding_box(
            annotated,
            candidate.x, candidate.y, candidate.w, candidate.h,
            result.label, result.confidence, color_bgr,
        )

        confidence_sum += result.confidence
        detections.append({
            "label": result.label,
            "confidence": round(result.confidence * 100, 2),
            "color": CLASS_COLORS_HEX[result.label],
            "box": {"x": candidate.x, "y": candidate.y, "w": candidate.w, "h": candidate.h},
            "probabilities": {
                cls: round(prob * 100, 2) for cls, prob in result.class_probabilities.items()
            },
        })

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    summary = {cls: 0 for cls in CLASS_NAMES}
    for detection in detections:
        summary[detection["label"]] += 1

    average_confidence = round(confidence_sum / len(detections) * 100, 2) if detections else 0.0

    return {
        "annotated_image": annotated,
        "detections": detections,
        "summary": summary,
        "leaf_count": len(detections),
        "average_confidence": average_confidence,
        "prediction_time_ms": elapsed_ms,
    }


def save_result_image(image_bgr: np.ndarray) -> str:
    """Menyimpan gambar hasil anotasi ke static/results dan mengembalikan URL relatifnya."""
    filename = f"result_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config["RESULT_FOLDER"], filename)
    cv2.imwrite(filepath, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return f"/static/results/{filename}"


@app.route("/")
def index():
    return render_template("index.html", model_ready=predictor.is_ready)


@app.route("/api/status")
def status():
    return jsonify({
        "model_ready": predictor.is_ready,
        "error": predictor.load_error,
        "classes": CLASS_NAMES,
    })


@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        if not predictor.is_ready:
            return jsonify({
                "success": False,
                "message": predictor.load_error or "Model belum siap digunakan.",
            }), 503

        image_bgr = None

        # Sumber 1: file upload (drag & drop / klik upload)
        if "image" in request.files and request.files["image"].filename != "":
            file = request.files["image"]
            if not allowed_file(file.filename):
                return jsonify({
                    "success": False,
                    "message": "Format file tidak didukung. Gunakan PNG, JPG, JPEG, atau WEBP.",
                }), 400
            secure_filename(file.filename)  # validasi nama file
            image_bgr = decode_upload_file(file)

        # Sumber 2: capture kamera (base64 data URL)
        elif request.is_json and request.get_json().get("image_base64"):
            image_bgr = decode_base64_image(request.get_json()["image_base64"])

        if image_bgr is None:
            return jsonify({
                "success": False,
                "message": "Tidak ada gambar yang diterima. Silakan upload atau capture gambar.",
            }), 400

        pipeline_result = run_detection_pipeline(image_bgr)
        result_url = save_result_image(pipeline_result["annotated_image"])

        return jsonify({
            "success": True,
            "result_image_url": result_url,
            "leaf_count": pipeline_result["leaf_count"],
            "summary": pipeline_result["summary"],
            "average_confidence": pipeline_result["average_confidence"],
            "prediction_time_ms": pipeline_result["prediction_time_ms"],
            "detections": pipeline_result["detections"],
            "disease_info": DISEASE_INFO,
        })

    except ModelNotLoadedError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 - fallback aman untuk seluruh error tak terduga
        app.logger.exception("Terjadi kesalahan pada proses deteksi")
        return jsonify({
            "success": False,
            "message": f"Terjadi kesalahan pada server: {exc}",
        }), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "success": False,
        "message": "Ukuran file melebihi batas maksimal 20 MB.",
    }), 413


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
