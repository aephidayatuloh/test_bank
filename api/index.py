import joblib
import pandas as pd
#import numpy as np
import json
import os

# 💡 Muat model di luar handler agar hanya dimuat sekali (cold start)
try:
    # Path relatif ke file joblib di direktori 'api/'
    model_path = os.path.join(os.path.dirname(__file__), 'random_forest_bank_marketing_pipline.joblib')
    MODEL_PIPELINE = joblib.load(model_path)
    print("Model Pipeline berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    MODEL_PIPELINE = None


def handler(request):
    """
    Fungsi handler untuk Vercel Serverless Function.
    """
    if MODEL_PIPELINE is None:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Model failed to load.'})
        }
    
    # 1. Ambil data dari request
    try:
        if request.method != 'POST':
            return {
                'statusCode': 405,
                'body': json.dumps({'message': 'Method Not Allowed. Use POST.'})
            }
            
        data = request.get_json()
        
        # 2. Konversi data input (list/dict) menjadi DataFrame
        input_df = pd.DataFrame(data['features'], index=[0])

    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Invalid input data: {e}'})
        }

    # 3. Prediksi menggunakan pipeline (preprocessor + classifier)
    try:
        prediction = MODEL_PIPELINE.predict(input_df)[0]
        prediction_proba = MODEL_PIPELINE.predict_proba(input_df)[0].tolist()

        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': int(prediction),
                'probability': prediction_proba
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Prediction failed: {e}'})
        }
