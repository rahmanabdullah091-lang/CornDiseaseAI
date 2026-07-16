# CornDiseaseAI

Website AI untuk identifikasi penyakit daun jagung menggunakan model
**Custom Convolutional Neural Network (CNN)** yang dikombinasikan dengan
sistem **pseudo-object detection** berbasis OpenCV.

Dibangun sebagai implementasi penelitian skripsi — single page application,
tanpa login, tanpa database.

## Model

| Item | Keterangan |
|---|---|
| File model | `model/BEST_MODEL_SIZE_156_SPLIT_20.h5` |
| Framework | TensorFlow / Keras |
| Input size | 156 x 156 |
| Akurasi | 98.38% |
| Tipe | Image Classification (bukan object detection) |
| Kelas | Healthy, Common Rust, Gray Leaf Spot, Northern Leaf Blight |

## Cara Kerja Sistem

```
User Upload / Capture Image
        ↓
OpenCV: Gaussian Blur → HSV → Threshold → Morphology
        ↓
Contour Detection (tiap kontur = satu daun)
        ↓
Crop tiap daun → Resize 156x156 → Normalisasi
        ↓
Prediksi dengan BEST_MODEL_SIZE_156_SPLIT_20.h5
        ↓
Gambar Bounding Box + Label + Confidence
        ↓
Tampilkan hasil di halaman web
```

## Struktur Project

```
CornDiseaseAI/
├── app.py                  # Entry point Flask
├── requirements.txt
├── model/
│   └── BEST_MODEL_SIZE_156_SPLIT_20.h5   (letakkan file model Anda di sini)
├── utils/
│   ├── __init__.py         # Konstanta: nama kelas, ukuran gambar, warna
│   ├── detector.py         # Segmentasi & pseudo-object detection (OpenCV)
│   ├── predictor.py        # Wrapper inference model CNN
│   └── preprocess.py       # Resize & normalisasi gambar
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   ├── uploads/             # (opsional, tidak wajib dipakai backend saat ini)
│   └── results/             # Gambar hasil anotasi bounding box disimpan di sini
├── templates/
│   └── index.html
└── README.md
```

## Instalasi

1. **Clone / buka project ini di VS Code.**

2. **Buat virtual environment (disarankan):**

   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS / Linux
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Letakkan file model Anda** di:

   ```
   model/BEST_MODEL_SIZE_156_SPLIT_20.h5
   ```

5. **Jalankan aplikasi:**

   ```bash
   python app.py
   ```

6. Buka browser ke `http://127.0.0.1:5000`

## Catatan Penting

- Jika file model belum diletakkan di folder `model/`, aplikasi tetap
  berjalan tetapi endpoint `/api/detect` akan mengembalikan pesan error
  yang jelas (status 503) sampai file model tersedia.
- Urutan kelas pada `utils/__init__.py` (`CLASS_NAMES`) **harus** sama
  persis dengan urutan `class_indices` yang digunakan saat training model.
  Jika model Anda dilatih dengan urutan alfabetis berbeda, sesuaikan
  urutan list tersebut.
- Sistem segmentasi OpenCV pada `utils/detector.py` menggunakan rentang
  warna HSV umum untuk daun jagung. Jika hasil segmentasi kurang akurat
  pada dataset foto Anda (misalnya karena pencahayaan/background berbeda),
  sesuaikan `LOWER_LEAF_HSV`, `UPPER_LEAF_HSV`, dan `MIN_AREA_RATIO` di
  file tersebut.
- Jika tidak ada kontur daun valid yang terdeteksi, sistem otomatis
  fallback dengan mengklasifikasikan seluruh gambar sebagai satu objek.

## Teknologi

- **Backend:** Flask, OpenCV (opencv-python-headless), TensorFlow/Keras, NumPy
- **Frontend:** HTML5, CSS3, JavaScript ES6 murni (tanpa framework/CSS library)
- **Font:** Inter, Poppins
- **Icon:** Font Awesome
