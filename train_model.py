"""
Modeli eğitip kaydetmek için bu scripti çalıştır:
    python train_model.py

Çıktı: heart_disease_model.pkl  (model + scaler birlikte)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# ─── Veri Yükleme ─────────────────────────────────────────────────────────────
# train.csv veya train.zip'ten okuyabilirsin.
# Kaggle'dan indirdiğin CSV dosyasını buraya ver:
TRAIN_PATH = "train.csv"   # ← gerekirse "train.zip" ile değiştir

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"'{TRAIN_PATH}' bulunamadı. Kaggle'dan train.csv'yi bu klasöre koy."
    )

train = pd.read_csv(TRAIN_PATH)
print(f"Veri boyutu: {train.shape}")

# ─── Feature Mühendisliği ─────────────────────────────────────────────────────
FEATURE_COLS = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
    "FBS over 120", "EKG results", "Max HR", "Exercise angina",
    "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium"
]

X = train[FEATURE_COLS]
y = (train["Heart Disease"] == "Presence").astype(int)   # 1=Presence, 0=Absence

# ─── Train / Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Scaler ───────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── Model Seçimi ─────────────────────────────────────────────────────────────
models = {
    "Logistic Regression":      LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":            RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":        GradientBoostingClassifier(n_estimators=100, random_state=42),
}

best_name, best_model, best_acc = None, None, 0.0

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_sc))
    print(f"  {name:30s} → Test Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc, best_name, best_model = acc, name, model

print(f"\n✅ En iyi model: {best_name}  (Accuracy: {best_acc:.4f})")
print(classification_report(y_test, best_model.predict(X_test_sc),
                             target_names=["Absence", "Presence"]))

# ─── Kaydet ───────────────────────────────────────────────────────────────────
save_obj = {"model": best_model, "scaler": scaler, "features": FEATURE_COLS}
with open("heart_disease_model.pkl", "wb") as f:
    pickle.dump(save_obj, f)

print("💾 Model kaydedildi: heart_disease_model.pkl")