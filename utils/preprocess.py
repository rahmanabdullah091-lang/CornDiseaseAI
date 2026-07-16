"""
preprocess.py

Modul preprocessing gambar sebelum masuk ke model Custom CNN.
Bertanggung jawab untuk:
    1. Resize gambar ke ukuran input model (156x156)
    2. Menyiapkan nilai pixel sebagai float32 rentang [0, 255]
       (model sudah punya layer Rescaling internal, lihat normalize_image)
    3. Konversi channel BGR (OpenCV) -> RGB (TensorFlow)
    4. Menambahkan batch dimension agar siap di-predict

Semua fungsi di sini bersifat pure (tidak menyimpan state) sehingga
mudah diuji dan digunakan ulang di modul lain (detector.py, predictor.py).
"""

from __future__ import annotations

import cv2
import numpy as np

from utils import IMAGE_SIZE


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Konversi array gambar dari format BGR (OpenCV) ke RGB (TensorFlow)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize gambar ke ukuran persegi (size x size) TANPA mempertahankan
    aspect ratio (akan meregangkan/menyempitkan gambar non-persegi).

    Disimpan sebagai utilitas umum, namun pipeline utama (preprocess_crop)
    TIDAK memakai fungsi ini lagi — lihat resize_with_padding() di bawah.
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def resize_with_padding(
    image: np.ndarray,
    size: int = IMAGE_SIZE,
    pad_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Resize gambar ke (size x size) SAMBIL mempertahankan aspect ratio asli,
    menambahkan padding (letterbox) di sisi yang lebih pendek agar tidak
    terjadi distorsi bentuk daun.

    Ini penting karena gambar upload pengguna bisa berukuran & rasio apa
    pun (mis. 387x516 potret vs 256x256 persegi). Resize langsung ke
    ukuran persegi tanpa mempertahankan rasio akan meregangkan daun secara
    tidak alami, menyebabkan model salah membaca bentuk gejala penyakit.

    Args:
        image: Gambar input (format bebas channel, biasanya RGB/BGR).
        size: Ukuran target sisi persegi.
        pad_color: Warna padding (default putih, netral untuk foto daun
            dengan latar terang; sesuaikan bila background foto training
            didominasi warna lain).

    Returns:
        Array gambar berukuran (size, size, channel) dengan konten asli
        di tengah dan proporsi bentuk aslinya terjaga.
    """
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    scale = size / max(height, width)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size, image.shape[2]), pad_color, dtype=image.dtype)
    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

    return canvas


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Menyiapkan nilai pixel sebagai float32 pada rentang asli [0, 255].

    PENTING: BEST_MODEL_SIZE_156_SPLIT_20.h5 sudah memiliki layer
    Rescaling(1./255) internal di dalam arsitekturnya (terkonfirmasi via
    diagnose_model.py: skema tanpa pembagian 255 menghasilkan prediksi
    yang bervariasi sesuai isi gambar, sedangkan skema rescale_0_1 dan
    centered_-1_1 membuat model 'buta' dan selalu condong ke satu kelas
    yang sama). Karena itu di sini kita TIDAK membagi dengan 255 lagi —
    cukup cast ke float32 agar tidak double-normalisasi.
    """
    return image.astype("float32")


def add_batch_dimension(image: np.ndarray) -> np.ndarray:
    """Menambahkan dimensi batch sehingga bentuk array menjadi (1, H, W, 3)."""
    return np.expand_dims(image, axis=0)


def preprocess_crop(crop_bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Pipeline lengkap preprocessing satu crop daun untuk siap diprediksi.

    Args:
        crop_bgr: Array gambar hasil crop (format BGR, ukuran bebas).
        size: Ukuran target sisi persegi (default mengikuti IMAGE_SIZE model).

    Returns:
        Array numpy berbentuk (1, size, size, 3) dengan nilai float32 [0, 255]
        (model memiliki layer Rescaling internal), siap diberikan ke
        model.predict().
    """
    rgb = bgr_to_rgb(crop_bgr)
    resized = resize_with_padding(rgb, size)
    normalized = normalize_image(resized)
    batched = add_batch_dimension(normalized)
    return batched


def preprocess_full_image(image_bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Preprocessing untuk gambar penuh (dipakai sebagai fallback jika
    tidak ada contour daun yang berhasil terdeteksi oleh detector.py).
    """
    return preprocess_crop(image_bgr, size)
