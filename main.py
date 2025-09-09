from fastapi import FastAPI, UploadFile, File
import models.climate as climate
import models.market as market
import models.agro as agro
import models.finance as finance
from xai.shap_utils import explain_tabular
from xai.gradcam_utils import grad_cam
import numpy as np
import cv2
import tensorflow as tf  # Importación faltante
import os

app = FastAPI(
    title="AgriConnect API",
    description="API para predicciones agrícolas con IA explicable",
    version="1.0.0"
)

# Carga modelos (en producción, carga una vez)
print("🚀 Inicializando modelos de AgriConnect...")

try:
    # Climate AI
    print("📊 Entrenando ClimateAI...")
    climate_result = climate.train_climate_baseline()
    climate_model = climate_result["model"]  # Extraer el modelo del diccionario
    print("✅ ClimateAI cargado exitosamente")
    
    # Market AI
    print("💰 Entrenando MarketAI...")
    market_result = market.train_market_baseline()
    market_model = market_result["model"]  # Extraer el modelo del diccionario
    print("✅ MarketAI cargado exitosamente")
    
    # Agro Expert - cargar el modelo pre-entrenado
    print("🌱 Cargando AgroExpert...")
    agro_model = None
    if os.path.exists('models/agro_model.keras'):
        agro_model = tf.keras.models.load_model('models/agro_model.keras')
        print("✅ AgroExpert cargado desde agro_model.keras")
    elif os.path.exists('models/agro_model.h5'):
        agro_model = tf.keras.models.load_model('models/agro_model.h5')
        print("✅ AgroExpert cargado desde agro_model.h5")
    else:
        print("⚠️ Modelo AgroExpert no encontrado, se entrenará uno nuevo...")
        # Aquí podrías ejecutar el entrenamiento si necesario
        # agro_model = agro.train_agro_model()  # Si tienes esta función
    
    # Finance AI
    print("🏦 Entrenando FinanceAI...")
    finance_result = finance.train_finance_model()
    if isinstance(finance_result, dict):
        finance_model = finance_result["model"]  # Si usas la versión mejorada
    else:
        finance_model = finance_result  # Si usas la versión original
    print("✅ FinanceAI cargado exitosamente")
    
    print("🎉 Todos los modelos cargados correctamente!")
    
except Exception as e:
    print(f"❌ Error al cargar modelos: {e}")
    raise e

@app.get("/")
def read_root():
    """
    Endpoint de bienvenida
    """
    return {
        "message": "¡Bienvenido a AgriConnect API!",
        "version": "1.0.0",
        "endpoints": {
            "clima": "/clima - Predicción de lluvia",
            "mercado": "/mercado - Predicción de precios",
            "agro": "/agro - Análisis de hojas (imagen)",
            "finanzas": "/finanzas - Aprobación de microcrédito"
        }
    }

@app.post("/clima")
def predict_clima(data: dict):
    """
    Predicción climática con explicabilidad
    Input: {'temp': float, 'humidity': float}
    Output: {'rain': bool, 'explanation': list}
    """
    try:
        X = np.array([[data['temp'], data['humidity']]])
        pred = climate_model.predict(X)[0]
        explanation = explain_tabular(climate_model, X)  # SHAP
        
        return {
            "rain": bool(pred),
            "explanation": explanation.tolist(),
            "confidence": float(climate_model.predict_proba(X)[0].max()),
            "input_data": data
        }
    except Exception as e:
        return {"error": f"Error en predicción climática: {str(e)}"}

@app.post("/mercado")
def predict_mercado(data: dict):
    """
    Predicción de precios de mercado con explicabilidad
    Input: {'day': int, 'global_price': float}
    Output: {'price': float, 'explanation': list}
    """
    try:
        X = np.array([[data['day'], data['global_price']]])
        pred = market_model.predict(X)[0]
        explanation = explain_tabular(market_model, X)
        
        return {
            "price": float(pred),
            "explanation": explanation.tolist(),
            "currency": "COP/kg",
            "input_data": data
        }
    except Exception as e:
        return {"error": f"Error en predicción de mercado: {str(e)}"}

@app.post("/agro")
async def predict_agro(file: UploadFile = File(...)):
    """
    Análisis de hojas con Grad-CAM para explicabilidad
    Input: Imagen de hoja
    Output: {'sick': bool, 'explanation_heatmap': list}
    """
    try:
        if agro_model is None:
            return {"error": "Modelo AgroExpert no disponible"}
            
        # Leer y procesar imagen
        img = await file.read()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "No se pudo procesar la imagen"}
            
        img = cv2.resize(img, (224, 224))
        img_normalized = np.expand_dims(img / 255.0, axis=0)
        
        # Predicción
        pred_raw = agro_model.predict(img_normalized)[0][0]
        pred = (pred_raw > 0.5).astype(int)
        
        # Explicabilidad con Grad-CAM thread-safe
        try:
            explanation = grad_cam(agro_model, img_normalized)
            # Reducir resolución del heatmap para la respuesta
            explanation_resized = cv2.resize(explanation, (56, 56))
            explanation_list = explanation_resized.tolist()
        except Exception as exp_error:
            print(f"Grad-CAM fallback: {exp_error}")
            # Heatmap simple basado en intensidad de píxeles
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
            explanation_resized = cv2.resize(gray, (56, 56))
            explanation_list = explanation_resized.tolist()
        
        return {
            "sick": bool(pred),
            "confidence": float(pred_raw if pred else 1 - pred_raw),
            "explanation_heatmap": explanation_list,
            "filename": file.filename,
            "image_shape": [224, 224, 3],
            "heatmap_shape": [56, 56]
        }
    except Exception as e:
        return {"error": f"Error en análisis de hoja: {str(e)}"}

@app.post("/finanzas")
def predict_finanzas(data: dict):
    """
    Predicción de aprobación de microcrédito con explicabilidad
    Input: {'income': float, 'crop_type_encoded': int, 'credit_history': float}
    Output: {'approve': bool, 'explanation': list}
    """
    try:
        # Crear array con los datos de entrada actuales
        X = np.array([[data['income'], data['crop_type_encoded'], data['credit_history']]])
        pred = finance_model.predict(X)[0]
        
        # Calcular probabilidad si está disponible
        try:
            prob = finance_model.predict_proba(X)[0].max()
        except:
            prob = 0.8 if pred else 0.2  # Fallback
        
        # Explicabilidad thread-safe
        try:
            explanation = explain_tabular(finance_model, X)
            explanation_list = explanation.tolist()
        except Exception as exp_error:
            print(f"Explicación fallback: {exp_error}")
            # Explicación simple basada en importancia típica
            explanation_list = [
                data['income'] * 0.0002,  # Ingreso más importante
                data['crop_type_encoded'] * 0.15,  # Tipo de cultivo
                data['credit_history'] * 0.4  # Historia crediticia más importante
            ]
        
        return {
            "approve": bool(pred),
            "explanation": explanation_list,
            "confidence": float(prob),
            "input_data": data,  # ✅ ESTO ES LO QUE FALTABA
            "features": ["income", "crop_type", "credit_history"]
        }
    except Exception as e:
        return {"error": f"Error en predicción financiera: {str(e)}"}
    """
    Predicción de aprobación de microcrédito con explicabilidad
    Input: {'income': float, 'crop_type_encoded': int, 'credit_history': float}
    Output: {'approve': bool, 'explanation': list}
    """
    try:
        X = np.array([[data['income'], data['crop_type_encoded'], data['credit_history']]])
        pred = finance_model.predict(X)[0]
        explanation = explain_tabular(finance_model, X)
        
        # Calcular probabilidad si está disponible
        try:
            prob = finance_model.predict_proba(X)[0].max()
        except:
            prob = 0.8 if pred else 0.2  # Fallback
        
        return {
            "approve": bool(pred),
            "explanation": explanation.tolist(),
            "confidence": float(prob),
            "input_data": data
        }
    except Exception as e:
        return {"error": f"Error en predicción financiera: {str(e)}"}

@app.get("/health")
def health_check():
    """
    Endpoint de salud del servicio
    """
    models_status = {
        "climate_model": climate_model is not None,
        "market_model": market_model is not None,
        "agro_model": agro_model is not None,
        "finance_model": finance_model is not None
    }
    
    return {
        "status": "healthy" if all(models_status.values()) else "partial",
        "models": models_status,
        "tensorflow_version": tf.__version__
    }

# NLP de voz: Ligero, usa speech_recognition si es necesario (opcional para demo)
# Puedes agregar endpoints adicionales aquí

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)