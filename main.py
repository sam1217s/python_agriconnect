from fastapi import FastAPI, UploadFile, File
import models.climate as climate
import models.market as market
import models.agro as agro
import models.finance as finance
from xai.shap_utils import explain_tabular
from xai.gradcam_utils import grad_cam
import numpy as np
import cv2

app = FastAPI()

# Carga modelos (en producción, carga una vez)
climate_model = climate.train_climate_baseline()
market_model = market.train_market_baseline()
agro_model = tf.keras.models.load_model('models/agro_model.h5')
finance_model = finance.train_finance_model()

@app.post("/clima")
def predict_clima(data: dict):  # Input: {'temp': float, 'humidity': float}
    X = np.array([[data['temp'], data['humidity']]])
    pred = climate_model.predict(X)[0]
    explanation = explain_tabular(climate_model, X)  # SHAP
    return {"rain": bool(pred), "explanation": explanation.tolist()}

@app.post("/mercado")
def predict_mercado(data: dict):  # {'day': int, 'global_price': float}
    X = np.array([[data['day'], data['global_price']]])
    pred = market_model.predict(X)[0]
    explanation = explain_tabular(market_model, X)
    return {"price": pred, "explanation": explanation.tolist()}

@app.post("/agro")
async def predict_agro(file: UploadFile = File(...)):
    img = await file.read()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img / 255.0, axis=0)
    pred = (agro_model.predict(img) > 0.5).astype(int)[0][0]
    explanation = grad_cam(agro_model, img)
    return {"sick": bool(pred), "explanation_heatmap": explanation.tolist()}

@app.post("/finanzas")
def predict_finanzas(data: dict):  # {'income': float, 'crop_type_encoded': int, 'credit_history': float}
    X = np.array([[data['income'], data['crop_type_encoded'], data['credit_history']]])
    pred = finance_model.predict(X)[0]
    explanation = explain_tabular(finance_model, X)
    return {"approve": bool(pred), "explanation": explanation.tolist()}

# NLP de voz: Ligero, usa speech_recognition si es necesario (opcional para demo)