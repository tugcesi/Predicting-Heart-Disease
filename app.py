import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ─── Sayfa Ayarları ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
)

MODEL_PATH = "heart_disease_model.pkl"

# ─── Model Yükleme ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

obj = load_model()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ Hakkında")
    st.markdown("""
    Bu uygulama, **Machine Learning** algoritmaları kullanarak
    kalp hastalığı riskini tahmin eder.

    **Kullanılan Özellikler:**
    - Yaş, Cinsiyet, Göğüs Ağrısı Tipi
    - Kan Basıncı, Kolesterol
    - EKG Sonuçları, Maks. Kalp Hızı
    - Egzersiz Anjini, ST Depresyonu
    - Fluro Damar Sayısı, Tallyum vb.

    **Model:** Scikit-learn (en iyi skor)
    """)
    st.markdown("---")
    st.markdown("**Geliştirici:** [tugcesi](https://github.com/tugcesi)")

# ─── Başlık ───────────────────────────────────────────────────────────────────
st.title("❤️ Heart Disease Prediction")
st.markdown("Hasta bilgilerini girerek kalp hastalığı riskini tahmin edin.")
st.markdown("---")

# ─── Model Uyarısı ────────────────────────────────────────────────────────────
if obj is None:
    st.error(
        "⚠️ **Model dosyası bulunamadı!**\n\n"
        "`train_model.py` scriptini çalıştırarak modeli oluştur:\n\n"
        "```bash\npython train_model.py\n```"
    )
    st.stop()

model   = obj["model"]
scaler  = obj["scaler"]

# ─── Giriş Formu ──────────────────────────────────────────────────────────────
st.subheader("🩺 Hasta Bilgileri")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Yaş", min_value=29, max_value=77, value=54, step=1)
    sex = st.selectbox("Cinsiyet", options=[("Erkek", 1), ("Kadın", 0)], format_func=lambda x: x[0])[1]
    chest_pain = st.selectbox(
        "Göğüs Ağrısı Tipi",
        options=[(1, "Tip 1 — Tipik Angina"),
                 (2, "Tip 2 — Atipik Angina"),
                 (3, "Tip 3 — Angina Dışı"),
                 (4, "Tip 4 — Asemptomatik")],
        format_func=lambda x: x[1]
    )[0]
    bp = st.number_input("Kan Basıncı (mmHg)", min_value=94, max_value=200, value=130, step=1)
    cholesterol = st.number_input("Kolesterol (mg/dl)", min_value=126, max_value=564, value=245, step=1)

with col2:
    fbs = st.selectbox("Açlık Kan Şekeri > 120", options=[("Hayır (0)", 0), ("Evet (1)", 1)],
                       format_func=lambda x: x[0])[1]
    ekg = st.selectbox(
        "EKG Sonucu",
        options=[(0, "Normal"), (1, "ST-T Dalgası Anormalliği"), (2, "Sol Ventrikül Hipertrofisi")],
        format_func=lambda x: x[1]
    )[0]
    max_hr = st.number_input("Maks. Kalp Hızı", min_value=71, max_value=202, value=153, step=1)
    exercise_angina = st.selectbox("Egzersiz Anjini", options=[("Hayır (0)", 0), ("Evet (1)", 1)],
                                   format_func=lambda x: x[0])[1]

with col3:
    st_depression = st.number_input("ST Depresyonu", min_value=0.0, max_value=6.2, value=0.5, step=0.1)
    slope_st = st.selectbox(
        "ST Eğimi",
        options=[(1, "Yukarı (1)"), (2, "Düz (2)"), (3, "Aşağı (3)")],
        format_func=lambda x: x[1]
    )[0]
    num_vessels = st.selectbox("Fluro Damar Sayısı", options=[0, 1, 2, 3])
    thallium = st.selectbox(
        "Talyum Testi",
        options=[(3, "Normal (3)"), (6, "Sabit Defekt (6)"), (7, "Geri Dönüşümlü Defekt (7)")],
        format_func=lambda x: x[1]
    )[0]

# ─── Tahmin ───────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🔍 Tahmin Et", use_container_width=True):
    input_data = np.array([[
        age, sex, chest_pain, bp, cholesterol,
        fbs, ekg, max_hr, exercise_angina,
        st_depression, slope_st, num_vessels, thallium
    ]])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    proba        = model.predict_proba(input_scaled)[0]

    presence_prob = proba[1] * 100
    absence_prob  = proba[0] * 100

    st.markdown("### 📊 Tahmin Sonucu")

    if prediction == 1:
        st.error(f"### 🚨 Kalp Hastalığı Riski: **PRESENCE**")
        st.markdown(f"Model, bu hastada kalp hastalığı varlığını tahmin ediyor.")
    else:
        st.success(f"### ✅ Kalp Hastalığı Riski: **ABSENCE**")
        st.markdown(f"Model, bu hastada kalp hastalığı olmadığını tahmin ediyor.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("🔴 Presence Olasılığı", f"%{presence_prob:.1f}")
        st.progress(presence_prob / 100)
    with col_b:
        st.metric("🟢 Absence Olasılığı", f"%{absence_prob:.1f}")
        st.progress(absence_prob / 100)

    st.markdown("---")
    st.caption("⚠️ Bu uygulama yalnızca eğitim amaçlıdır. Tıbbi karar için uzman hekiminize danışınız.")

st.markdown("---")
st.caption("Powered by Scikit-learn & Streamlit | [GitHub](https://github.com/tugcesi/Predicting-Heart-Disease)")