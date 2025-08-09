import streamlit as st
import re
import string
import joblib
import numpy as np


# preprocessing

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # hapus URL
    text = text.translate(str.maketrans('', '', string.punctuation))  # hapus tanda baca
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Load model & vectorizer

@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load("fake_news_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Gagal memuat model/vectorizer: {e}")
        return None, None

model, vectorizer = load_model_and_vectorizer()


# UI

st.set_page_config(page_title="Deteksi Berita Palsu", layout="centered")

st.title("📰 Deteksi Berita Palsu")
st.caption("Masukkan judul dan/atau isi berita dalam bahasa Inggris, lalu tekan **Cek Berita**.")

title = st.text_input("Judul Berita", "")
text = st.text_area("Isi / Narasi Berita", height=200)

if st.button("Cek Berita"):
    if (title.strip() == "") and (text.strip() == ""):
        st.warning("Masukkan judul atau narasi berita terlebih dahulu.")
    elif model is None or vectorizer is None:
        st.error("File model dan tfidf_vectorizer.pkl tidak ditemukan.")
    else:
        # Gabung judul + isi berita
        full_text = f"{title} {text}"
        cleaned_text = clean_text(full_text)

        # Transform dan prediksi
        X = vectorizer.transform([cleaned_text])
        pred = model.predict(X)[0]

        # Cek confidence (jika tersedia)
        confidence_pct = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence_pct = float(np.max(proba)) * 100
            elif hasattr(model, "decision_function"):
                score = model.decision_function(X)
                if score.ndim == 1:
                    prob = 1 / (1 + np.exp(-score))[0]
                    confidence_pct = float(max(prob, 1-prob) * 100)
        except Exception:
            pass

        # Mapping label
        if isinstance(pred, str):
            label = pred.upper()
        else:
            label = "FAKE" if pred == 1 else "REAL"

        # Tampilkan hasil
        if label == "FAKE":
            st.error(f"⚠️ Dinyatakan PALSU ({label})")
        else:
            st.success(f"✅ Dinyatakan ASLI ({label})")

        if confidence_pct is not None:
            st.write(f"**Confidence:** {confidence_pct:.1f}%")

        # Top TF-IDF terms
        try:
            coo = X[0].tocoo()
            indices = coo.col
            data = coo.data
            if len(indices) > 0:
                feat_names = np.array(vectorizer.get_feature_names_out())
                top_n = min(10, len(indices))
                top_idx = np.argsort(data)[-top_n:][::-1]
                top_terms = feat_names[indices[top_idx]]
                st.write("**Top terms (TF-IDF):**", ", ".join(top_terms))
        except Exception:
            pass
