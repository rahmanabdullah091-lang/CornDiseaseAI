"""
detector.py

Karena model BEST_MODEL_SIZE_156_SPLIT_20.h5 adalah model klasifikasi
(image classification) dan bukan object detection, modul ini membangun
sistem "pseudo-object detection" menggunakan OpenCV murni:

    1. Gaussian Blur       -> mengurangi noise pada gambar
    2. Konversi ke HSV     -> mempermudah segmentasi warna daun
    3. Thresholding (HSV)  -> memisahkan area daun dari background
    4. Morphology (Open/Close) -> membersihkan noise & menutup lubang kecil
    5. Contour Detection   -> menemukan setiap objek daun sebagai kontur
    6. Bounding Box + Crop -> setiap kontur di-crop untuk diklasifikasikan
       satu per satu oleh predictor.py

Hasil dari modul ini adalah daftar kandidat daun (bounding box + gambar
crop) yang siap diteruskan ke pipeline klasifikasi CNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

# Rentang warna HSV untuk mendeteksi daun jagung (hijau - kekuningan).
# Rentang dibuat cukup lebar agar mampu menangkap daun yang sudah
# terinfeksi penyakit (warna kecoklatan / berbintik) tanpa kehilangan
# objek daun itu sendiri.
LOWER_LEAF_HSV = np.array([15, 25, 25])
UPPER_LEAF_HSV = np.array([95, 255, 255])

# Luas kontur minimum (dalam pixel) agar dianggap sebagai daun yang valid.
# Nilai ini relatif terhadap luas total gambar (lihat _min_contour_area).
MIN_AREA_RATIO = 0.012

# Padding tambahan (pixel) di sekitar bounding box hasil crop agar daun
# tidak terpotong terlalu ketat, memberi konteks lebih baik untuk CNN.
BOX_PADDING = 12


@dataclass
class LeafCandidate:
    """Representasi satu daun hasil segmentasi OpenCV."""

    x: int
    y: int
    w: int
    h: int
    crop_bgr: np.ndarray
    contour_area: float


def _min_contour_area(image_shape: tuple[int, int]) -> float:
    height, width = image_shape[:2]
    return (height * width) * MIN_AREA_RATIO


def _build_leaf_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Membangun binary mask area daun menggunakan Gaussian Blur + HSV threshold + Morphology."""

    # 1. Gaussian Blur untuk mengurangi noise sebelum segmentasi warna
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)

    # 2. Konversi ke HSV agar segmentasi warna lebih stabil terhadap pencahayaan
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 3. Thresholding berdasarkan rentang warna daun
    mask = cv2.inRange(hsv, LOWER_LEAF_HSV, UPPER_LEAF_HSV)

    # 4. Morphology: Opening untuk menghapus noise kecil, Closing untuk
    #    menutup lubang kecil di dalam area daun.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    return mask


def _extract_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_leaves(image_bgr: np.ndarray) -> List[LeafCandidate]:
    """Mendeteksi seluruh daun jagung pada sebuah gambar.

    Args:
        image_bgr: Gambar input dalam format BGR (hasil cv2.imread / cv2.imdecode).

    Returns:
        List LeafCandidate terurut dari kiri-atas ke kanan-bawah. Jika tidak
        ada kontur valid ditemukan, list akan kosong (pemanggil bertanggung
        jawab melakukan fallback ke seluruh gambar).
    """
    height, width = image_bgr.shape[:2]
    mask = _build_leaf_mask(image_bgr)
    contours = _extract_contours(mask)
    min_area = _min_contour_area(image_bgr.shape)

    candidates: List[LeafCandidate] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Tambahkan padding tanpa keluar dari batas gambar
        x0 = max(0, x - BOX_PADDING)
        y0 = max(0, y - BOX_PADDING)
        x1 = min(width, x + w + BOX_PADDING)
        y1 = min(height, y + h + BOX_PADDING)

        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        candidates.append(
            LeafCandidate(
                x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                crop_bgr=crop, contour_area=area,
            )
        )

    # Urutkan dari kiri-atas ke kanan-bawah agar hasil konsisten & mudah dibaca
    candidates.sort(key=lambda c: (c.y, c.x))

    return candidates


def draw_bounding_box(
    image_bgr: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    confidence: float,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    """Menggambar bounding box beserta label penyakit dan confidence pada gambar.

    Modifikasi dilakukan secara in-place pada `image_bgr` dan juga dikembalikan
    untuk kenyamanan chaining.
    """
    thickness = max(2, round(min(image_bgr.shape[:2]) / 250))

    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color_bgr, thickness)

    text = f"{label} {confidence * 100:.2f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(image_bgr.shape[:2]) / 900)
    text_thickness = max(1, thickness - 1)

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

    label_y0 = max(0, y - text_h - baseline - 6)
    cv2.rectangle(
        image_bgr,
        (x, label_y0),
        (x + text_w + 10, label_y0 + text_h + baseline + 6),
        color_bgr,
        thickness=-1,
    )
    cv2.putText(
        image_bgr,
        text,
        (x + 5, label_y0 + text_h + 2),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    return image_bgr
