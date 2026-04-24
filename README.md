# ❤️ Predicting Heart Disease with Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![License](https://img.shields.io/github/license/tugcesi/Predicting-Heart-Disease)

Çeşitli ML sınıflandırma algoritmalarını kullanarak kalp hastalığı riskini tahmin eden bir makine öğrenmesi projesi.

---

## 📌 Proje Özeti

Bu projede [Kaggle Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2) veri seti kullanılmıştır.
630.000 eğitim kaydı üzerinde birden fazla ML algoritması karşılaştırılmış; en iyi model Streamlit arayüzüne entegre edilmiştir.

**Hedef Değişken:** `Heart Disease` → `Presence` / `Absence`

---

## 📊 Özellikler (Features)

| Özellik | Açıklama |
|---------|----------|
| Age | Yaş |
| Sex | Cinsiyet (0: Kadın, 1: Erkek) |
| Chest pain type | Göğüs ağrısı tipi (1–4) |
| BP | Kan basıncı (mmHg) |
| Cholesterol | Kolesterol (mg/dl) |
| FBS over 120 | Açlık kan şekeri > 120 (0/1) |
| EKG results | EKG sonucu (0/1/2) |
| Max HR | Maksimum kalp hızı |
| Exercise angina | Egzersiz anjini (0/1) |
| ST depression | ST depresyonu |
| Slope of ST | ST eğimi (1/2/3) |
| Number of vessels fluro | Fluro ile görülen damar sayısı (0–3) |
| Thallium | Tallyum testi (3/6/7) |

---

## 🤖 Kullanılan Algoritmalar

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest ✅
- AdaBoost
- Gradient Boosting
- Gaussian Naive Bayes
- XGBoost
- LightGBM
- CatBoost

---

## 🚀 Kurulum & Çalıştırma

### 1. Repoyu Klonla
```bash
git clone https://github.com/tugcesi/Predicting-Heart-Disease.git
cd Predicting-Heart-Disease
```

### 2. Gereksinimleri Yükle
```bash
pip install -r requirements.txt
```

### 3. Modeli Eğit & Kaydet
```bash
python train_model.py
```
> `heart_disease_model.pkl` dosyası oluşturulacaktır.

### 4. Streamlit Uygulamasını Başlat
```bash
streamlit run app.py
```

Tarayıcında `http://localhost:8501` adresini aç.

---

## 📁 Dosya Yapısı

```
Predicting-Heart-Disease/
├── app.py                              # Streamlit uygulaması
├── train_model.py                      # Model eğitim scripti
├── heart_disease_model.pkl             # Eğitilmiş model (train_model.py ile oluştur)
├── predicting-heart-disease-ML.ipynb   # Keşifsel analiz & model karşılaştırması
├── train.csv / train.zip               # Eğitim verisi (Kaggle'dan indir)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım |
|-----------|----------|
| Python | Temel dil |
| Scikit-learn | Model eğitimi & değerlendirme |
| XGBoost / LightGBM / CatBoost | Gelişmiş boosting modelleri |
| Streamlit | Web arayüzü |
| Pandas / NumPy | Veri işleme |

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
