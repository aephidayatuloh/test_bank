# api/index.py

import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from mangum import Mangum
import os

# 💡 Pustaka untuk download dari Hugging Face
from huggingface_hub import hf_hub_download 

# --- KONFIGURASI HF HUB ---
HF_REPO_ID = "aephidayatuloh/bank-model" # Ganti dengan repo Anda
HF_MODEL_FILENAME = "random_forest_bank_marketing_pipeline.joblib"

# --- 1. SETUP MODEL DAN PATH ---
try:
    # 💡 LAKUKAN DOWNLOAD MODEL DARI HF HUB
    downloaded_model_path = hf_hub_download(
        repo_id=HF_REPO_ID, 
        filename=HF_MODEL_FILENAME
    )
    
    # Muat model dari file yang baru diunduh (di cache Vercel)
    MODEL_PIPELINE = joblib.load(downloaded_model_path)
    
    # Ambil nama fitur (tetap diperlukan untuk DataFrame)
    # [Tambahkan logika pengambilan nama fitur Anda di sini, jika diperlukan]
    
    print("✅ Model Pipeline berhasil diunduh dan dimuat dari Hugging Face Hub.")

except Exception as e:
    print(f"❌ Gagal memuat model dari HF Hub atau joblib: {e}")
    MODEL_PIPELINE = None

# --- 2. DEFINISI STRUKTUR DATA (Pydantic Schema) ---

# Ganti dengan semua fitur yang dibutuhkan model Anda
class PredictionInput(BaseModel):
    """Skema Pydantic untuk validasi data input."""
    age: int
    balance: int
    day: int
    campaign: int
    pdays: int
    previous: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    poutcome: str

# --- 3. DEFINISI APLIKASI FASTAPI ---

app = FastAPI(
    title="Bank Deposit Prediction API",
    version="1.0.0",
    description="API untuk memprediksi probabilitas deposit berjangka menggunakan model Scikit-learn Pipeline yang di-deploy di Vercel."
)

@app.get("/", tags=["Health Check"])
def home():
    """Endpoint untuk health check."""
    return {"status": "ok", "message": "API is running."}

@app.post("/predict", tags=["Prediction"])
def predict(data: PredictionInput):
    """
    Melakukan prediksi deposit berdasarkan data pelanggan.
    Data akan ditransformasi otomatis oleh sklearn.Pipeline di server.
    """
    if MODEL_PIPELINE is None:
        raise HTTPException(status_code=500, detail="Model gagal dimuat di server.")

    try:
        # Konversi Pydantic model (data) ke dictionary, lalu ke DataFrame Pandas
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])

        # Prediksi
        prediction = MODEL_PIPELINE.predict(input_df)[0]
        prediction_proba = MODEL_PIPELINE.predict_proba(input_df)[0].tolist()

        return {
            "prediction_class": int(prediction),
            "probability": {
                "no_deposit": prediction_proba[0],
                "yes_deposit": prediction_proba[1]
            },
            "interpretation": "Class 1 = Yes (Deposit), Class 0 = No (Non-Deposit)"
        }
    except Exception as e:
        # Menangkap error saat prediksi (misal: mismatch fitur)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# --- 4. EXPORT HANDLER MANGUM ---

# Handler yang akan digunakan oleh Vercel
handler = Mangum(app)

# Setelah di-deploy ke Vercel, dokumentasi otomatis tersedia di:
# - Dokumentasi Swagger UI: /docs
# - Dokumentasi ReDoc: /redoc
