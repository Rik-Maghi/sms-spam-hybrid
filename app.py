import streamlit as st
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi SMS Spam Hybrid",
    page_icon="🛡️",
    layout="centered"
)

# --- LOAD MODEL & STEMMER DENGAN CACHE ---
# Fungsi cache agar model tidak perlu di-load berulang kali setiap kali tombol ditekan
@st.cache_resource
def load_model():
    return joblib.load('hybrid_spam_model.pkl')

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

try:
    model = load_model()
    stemmer = load_stemmer()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_msg = str(e)

# --- FUNGSI PREPROCESSING TEXT ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return stemmer.stem(text)

# --- DESAIN UI WEB ---
st.title("🛡️ Indonesian SMS Spam Detector")
st.markdown("""
Aplikasi ini menggunakan arsitektur **Hybrid Genetic Algorithm + Stacking Ensemble** untuk mengklasifikasikan pesan SMS ke dalam tiga kategori:
- 🟢 **Normal:** Pesan pribadi/wajar yang sah.
- 🔴 **Fraud:** Penipuan (Phishing, pencurian data, dll).
- 🟠 **Promo:** Iklan atau penawaran komersial.
""")

if not model_loaded:
    st.error(f"Gagal memuat model. Pastikan file 'hybrid_spam_model.pkl' berada di folder yang sama. Error: {error_msg}")
else:
    st.write("---")
    
    # Input dari user
    user_input = st.text_area(
        "📝 Masukkan teks SMS di sini:", 
        height=150, 
        placeholder="Contoh: Selamat! Anda memenangkan uang tunai 100jt, klik link berikut untuk klaim..."
    )
    
    # Tombol analisis
    if st.button("🔍 Analisis SMS", use_container_width=True):
        if not user_input.strip():
            st.warning("⚠️ Mohon masukkan teks SMS terlebih dahulu.")
        else:
            with st.spinner("⚙️ Menganalisis pola teks (Stemming & Feature Extraction)..."):
                # 1. Preprocessing (Membersihkan teks & Stemming)
                clean_text = preprocess_text(user_input)
                
                # 2. Melakukan Prediksi
                prediction = model.predict([clean_text])[0]
                
                # 3. Mengambil nilai Probabilitas (Tingkat Keyakinan)
                proba = model.predict_proba([clean_text])[0]
                normal_prob = proba[0] * 100
                fraud_prob = proba[1] * 100
                promo_prob = proba[2] * 100
                
                # 4. Menampilkan Hasil Visual
                st.write("---")
                st.write("### 📊 Hasil Analisis:")
                
                if prediction == 0:
                    st.success("🟢 **VONIS: SMS NORMAL**")
                    st.info("Pesan ini dinilai aman dan merupakan percakapan wajar.")
                elif prediction == 1:
                    st.error("🔴 **VONIS: SMS FRAUD (PENIPUAN)**")
                    st.warning("⚠️ HATI-HATI! Pesan ini terindikasi kuat sebagai penipuan. Jangan mengklik tautan (link) apa pun atau membagikan OTP.")
                elif prediction == 2:
                    st.warning("🟠 **VONIS: SMS PROMO**")
                    st.info("Pesan ini terdeteksi sebagai iklan, promosi, atau penawaran komersial (Spam Ringan).")
                    
                # Menampilkan Bar Persentase
                st.write("#### 🎯 Tingkat Keyakinan Model (Confidence Level):")
                st.progress(int(normal_prob), text=f"Normal: {normal_prob:.1f}%")
                st.progress(int(fraud_prob), text=f"Fraud (Penipuan): {fraud_prob:.1f}%")
                st.progress(int(promo_prob), text=f"Promo (Iklan): {promo_prob:.1f}%")
                
                st.write("---")
                with st.expander("Klik untuk melihat teks setelah di-stemming (Preprocessed Text)"):
                    st.code(clean_text)
