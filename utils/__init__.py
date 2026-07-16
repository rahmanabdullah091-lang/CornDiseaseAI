"""
CornDiseaseAI - utils package.

Berisi modul-modul inti sistem:
- preprocess.py -> normalisasi & resize gambar untuk model CNN
- detector.py   -> segmentasi & pseudo-object detection daun jagung (OpenCV)
- predictor.py  -> wrapper inference model Custom CNN (TensorFlow/Keras)
"""

# Nama kelas penyakit sesuai urutan output model.
# PENTING: urutan ini harus sama persis dengan urutan class_indices
# yang digunakan saat training model BEST_MODEL_SIZE_156_SPLIT_20.h5
#
# Urutan di bawah ini dikonfirmasi SECARA EMPIRIS menggunakan
# diagnose_model.py terhadap 4 gambar dengan label diketahui:
# index 0 -> Northern Leaf Blight, 1 -> Common Rust,
# index 2 -> Gray Leaf Spot,       3 -> Healthy
# (sesuai urutan alfabetis folder training: Blight, Common_Rust,
# Gray_Leaf_Spot, Healthy — urutan default ImageDataGenerator.flow_from_directory)
CLASS_NAMES = [
    "Northern Leaf Blight",
    "Common Rust",
    "Gray Leaf Spot",
    "Healthy",
]

# Ukuran input model (sesuai spesifikasi training)
IMAGE_SIZE = 156

# Warna bounding box per kelas (format BGR karena digambar dengan OpenCV)
CLASS_COLORS_BGR = {
    "Healthy": (76, 175, 80),               # Hijau
    "Common Rust": (54, 54, 220),           # Merah
    "Gray Leaf Spot": (220, 130, 30),       # Biru
    "Northern Leaf Blight": (0, 140, 255),  # Orange
}

# Warna bounding box untuk frontend (format HEX, dipakai pada respons JSON)
CLASS_COLORS_HEX = {
    "Healthy": "#4CAF50",
    "Common Rust": "#E53935",
    "Gray Leaf Spot": "#1E88E5",
    "Northern Leaf Blight": "#FB8C00",
}
