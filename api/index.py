# api/index.py

import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from mangum import Mangum
import os

# --- 1. SETUP MODEL DAN PATH ---

# Path ke file model (harus disesuaikan dengan struktur Vercel)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'random_forest_bank_marketing_pipeline.joblib')

try:
    # Muat model di luar handler/function untuk meminimalkan cold start
    MODEL_PIPELINE = joblib.load(MODEL_PATH)
    # Tentukan nama fitur (diambil dari langkah preprocessor jika perlu)
    FEATURE_NAMES = MODEL_PIPELINE.named_steps['preprocessor'].transformers_[0][2] + \
                    MODEL_PIPELINE.named_steps['preprocessor'].transformers_[1][2].tolist()
    print("✅ Model Pipeline berhasil dimuat.")
except Exception as e:
    # Log error model load, tetapi biarkan aplikasi berjalan (error 500 nanti)
    print(f"❌ Gagal memuat model: {e}")
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
    title="Bank Deposit Campaign Prediction API",
    version="1.0.0",
    description="API untuk memprediksi probabilitas membuka deposit berjangka menggunakan model Scikit-learn Pipeline yang di-deploy di Vercel."
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
