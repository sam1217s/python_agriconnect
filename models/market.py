import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('data/prices/sipsa_prices.csv')
X = df[['day', 'global_price']]
y = df['precio']  # COP/kg
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"MAE MarketAI: {mae:.2f} (precios COP/kg con SIPSA simulado semana 35 2025)")
joblib.dump(model, 'models/market_model.pkl')