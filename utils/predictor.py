"""
predictor.py

Wrapper untuk model Custom CNN (BEST_MODEL_SIZE_156_SPLIT_20.h5) yang
telah dilatih menggunakan TensorFlow/Keras.

Model dimuat sekali (singleton pattern) saat aplikasi Flask start,
sehingga proses inference selanjutnya menjadi cepat karena tidak perlu
memuat ulang bobot model pada setiap request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np

from utils import CLASS_NAMES, IMAGE_SIZE
from utils.preprocess import preprocess_crop

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "BEST_MODEL_SIZE_156_SPLIT_20.h5",
)


@dataclass
class PredictionResult:
    """Hasil prediksi untuk satu crop daun."""

    label: str
    confidence: float
    class_probabilities: dict


class ModelNotLoadedError(RuntimeError):
    """Dilempar ketika prediksi dicoba sebelum model berhasil dimuat."""


class CornDiseasePredictor:
    """Singleton wrapper untuk model klasifikasi penyakit daun jagung."""

    _instance: "CornDiseasePredictor | None" = None

    def __init__(self) -> None:
        self._model = None
        self._load_error: str | None = None
        self._load_model()

    @classmethod
    def get_instance(cls) -> "CornDiseasePredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> None:
        """Memuat model .h5 ke memori. Import TensorFlow dilakukan secara
        lazy (di dalam fungsi) agar startup aplikasi lebih cepat dan agar
        error TensorFlow tidak menghentikan proses import modul lain.
        """
        try:
            # Import lazy: TensorFlow cukup berat untuk di-load
            import tensorflow as tf

            if not os.path.exists(MODEL_PATH):
                self._load_error = (
                    f"File model tidak ditemukan di '{MODEL_PATH}'. "
                    "Pastikan BEST_MODEL_SIZE_156_SPLIT_20.h5 sudah "
                    "diletakkan pada folder 'model/'."
                )
                return

            # compile=False: kita hanya butuh arsitektur + bobot untuk
            # inference, bukan untuk melanjutkan training. Ini juga
            # menghindari error deserialisasi config optimizer/loss lama
            # (mis. `reduction="auto"`) yang tidak lagi dikenali oleh
            # versi Keras yang lebih baru dibanding saat model dilatih.
            self._model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as exc:  # noqa: BLE001 - ingin menangkap semua error load
            self._load_error = f"Gagal memuat model: {exc}"

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def predict(self, crop_bgr: np.ndarray) -> PredictionResult:
        """Melakukan prediksi kelas penyakit untuk satu gambar crop daun.

        Args:
            crop_bgr: Gambar crop daun (format BGR dari OpenCV).

        Returns:
            PredictionResult berisi label kelas dengan confidence tertinggi
            beserta seluruh distribusi probabilitas per kelas.

        Raises:
            ModelNotLoadedError: jika model belum berhasil dimuat.
        """
        if not self.is_ready:
            raise ModelNotLoadedError(
                self._load_error or "Model belum siap digunakan."
            )

        batch = preprocess_crop(crop_bgr, IMAGE_SIZE)
        raw_output = self._model.predict(batch, verbose=0)[0]

        probabilities = self._to_probability_dict(raw_output)
        best_index = int(np.argmax(raw_output))
        best_label = CLASS_NAMES[best_index]
        best_confidence = float(raw_output[best_index])

        return PredictionResult(
            label=best_label,
            confidence=best_confidence,
            class_probabilities=probabilities,
        )

    def predict_batch(self, crops_bgr: List[np.ndarray]) -> List[PredictionResult]:
        """Melakukan prediksi untuk banyak crop daun sekaligus."""
        return [self.predict(crop) for crop in crops_bgr]

    @staticmethod
    def _to_probability_dict(raw_output: np.ndarray) -> dict:
        return {
            CLASS_NAMES[i]: float(raw_output[i])
            for i in range(len(CLASS_NAMES))
        }
