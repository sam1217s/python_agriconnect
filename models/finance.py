import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import joblib  # Para guardar modelo

def load_finance_data():
    # Carga el CSV simulado (data/finance/finance_sim.csv)
    df = pd.read_csv('data/finance/finance_sim.csv')
    print("Datos cargados: ", df.shape)  # Verifica (1000, 4)
    print("Muestra:\n", df.head())
    return df

def train_finance_model():
    df = load_finance_data()
    X = df[['income', 'crop_type_encoded', 'credit_history']]
    y = df['approve']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    recall = recall_score(y_test, preds)
    print(f"Recall FinanceAI: {recall:.2f} (approve microcrédito con simulación Bancóldex/Finagro 2023-2025, minimiza rechazos erróneos)")
    joblib.dump(model, 'models/finance_model.pkl')  # Guarda para API
    return model

# Ejecuta el entrenamiento
train_finance_model()