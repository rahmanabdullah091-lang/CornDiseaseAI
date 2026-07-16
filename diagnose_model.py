"""
diagnose_model.py

Script diagnostik BERDIRI SENDIRI (tidak lewat Flask/OpenCV pipeline)
untuk mengecek apakah masalah "prediksi selalu sama" berasal dari:

    (A) Model itu sendiri / cara load
    (B) Urutan class_indices yang salah
    (C) Normalisasi preprocessing yang tidak sesuai dengan saat training

Cara pakai:
    python diagnose_model.py path/to/gambar1.jpg path/to/gambar2.jpg path/to/gambar3.jpg

Idealnya berikan 3-4 gambar dengan kelas BERBEDA yang Anda tahu label
aslinya (misalnya 1 foto daun sehat, 1 rust, 1 gray leaf spot, 1 blight).
"""

import sys
import os

import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "BEST_MODEL_SIZE_156_SPLIT_20.h5")
CLASS_NAMES_CURRENT = ["Northern Leaf Blight", "Common Rust", "Gray Leaf Spot", "Healthy"]
IMG_SIZE = 156


def load_and_prep_variants(path: str):
    """Menyiapkan gambar dengan BEBERAPA skema normalisasi berbeda,
    supaya kita bisa bandingkan mana yang menghasilkan distribusi
    probabilitas paling masuk akal (bukan selalu satu kelas dominan).
    """
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")

    variants = {
        "rescale_0_1 (bagi 255)": np.expand_dims(arr / 255.0, axis=0),
        "raw_0_255 (tanpa normalisasi)": np.expand_dims(arr, axis=0),
        "centered_-1_1 ((x/127.5)-1)": np.expand_dims((arr / 127.5) - 1.0, axis=0),
    }
    return variants


def main():
    if len(sys.argv) < 2:
        print("Gunakan: python diagnose_model.py gambar1.jpg gambar2.jpg ...")
        sys.exit(1)

    print(f"Memuat model dari: {MODEL_PATH}\n")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("=" * 70)
    print(f"Model input shape : {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print("=" * 70)
    print()

    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print(f"[SKIP] File tidak ditemukan: {path}")
            continue

        print(f"\n### Gambar: {os.path.basename(path)}")
        variants = load_and_prep_variants(path)

        for variant_name, batch in variants.items():
            preds = model.predict(batch, verbose=0)[0]
            best_idx = int(np.argmax(preds))
            best_label = CLASS_NAMES_CURRENT[best_idx] if best_idx < len(CLASS_NAMES_CURRENT) else f"index-{best_idx}"

            probs_str = ", ".join(
                f"{CLASS_NAMES_CURRENT[i]}={preds[i]*100:.2f}%"
                for i in range(len(preds))
            )
            print(f"  [{variant_name}]")
            print(f"    -> Prediksi teratas: {best_label} ({preds[best_idx]*100:.2f}%)")
            print(f"    -> Semua kelas     : {probs_str}")

    print("\n" + "=" * 70)
    print("CARA MEMBACA HASIL:")
    print("- Bandingkan 3 baris skema normalisasi untuk tiap gambar.")
    print("  Skema yang menghasilkan distribusi probabilitas paling")
    print("  'masuk akal' dan BERBEDA-BEDA antar gambar = skema yang benar.")
    print("- Jika SEMUA skema tetap menghasilkan kelas yang sama persis")
    print("  untuk gambar yang jelas berbeda, kemungkinan masalah ada pada")
    print("  urutan CLASS_NAMES (index benar tapi nama label salah) atau")
    print("  model itu sendiri (mis. bobot tidak ter-load dengan benar).")
    print("=" * 70)


if __name__ == "__main__":
    main()
