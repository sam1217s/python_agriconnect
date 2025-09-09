# models/climate.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

def train_climate_baseline():
    # Cargar dataset
    df = pd.read_csv('data/weather/historical_weather.csv')

    # Variables de entrada y salida
    X = df[['temp', 'humidity']]
    y = df['rain']

    # Dividir en train/test con balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modelo balanceado
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluación
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    print(f"F1-score ClimateAI: {f1:.2f} (rain/no rain con IDEAM simulado 2023-2025, balanceado)")

    # Guardar modelo
    joblib.dump(model, 'models/climate_model.pkl')

    # Retornar modelo y métrica (útil para FastAPI)
    return {"model": model, "f1_score": f1}
