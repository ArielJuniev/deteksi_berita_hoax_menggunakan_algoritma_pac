# Deteksi Berita Palsu (Fake News Detection)

Aplikasi berbasis web untuk mendeteksi apakah sebuah berita termasuk **FAKE** (palsu) atau **REAL** (asli) menggunakan model machine learning dengan algoritma Passive Aggressive Classifier dan TF-IDF.

---

## Fitur

- Deteksi berita palsu berdasarkan judul dan isi berita.
- Preprocessing teks otomatis (menghapus URL, tanda baca, dan case folding).
- Menampilkan confidence level hasil prediksi.
- Menampilkan top kata kunci berdasarkan TF-IDF dari berita yang dicek.

---

## Struktur Proyek

- `fakenewsdetection.py` — Script untuk melatih model, evaluasi, dan menyimpan model serta vectorizer.
- `app.py` — Aplikasi Streamlit untuk interaksi pengguna dan deteksi berita.
- `fake_news_model.pkl` — Model machine learning yang sudah dilatih.
- `tfidf_vectorizer.pkl` — Vectorizer TF-IDF yang sudah dilatih.
- `contoh-berita.csv` — Dataset contoh berisi berita dan label.

---

## Cara Instalasi

1. Clone repository ini:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

pip install -r requirements.txt

python fakenewsdetection.py

streamlit run app.py
