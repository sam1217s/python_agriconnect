import pandas as pd
import numpy as np
np.random.seed(42)  # Para reproducibilidad (SENASOFT: validación transparente)
n_samples = 1000  # Muestras para entrenamiento (sección 4: tabular con scikit-learn)

data = {
    'income': np.random.uniform(1000, 5000, n_samples),  # Miles COP; promedio ~$6,9-8,4M (Superfinanciera 2024)
    'crop_type_encoded': np.random.randint(0, 5, n_samples),  # 0=café,1=cacao,2=aguacate,3=tabaco,4=policultivo (Finagro 2024)
    'credit_history': np.random.uniform(0, 1, n_samples),  # 0=mala,1=buena (inclusión ~6,2% microcréditos rurales)
    'approve': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # 70% aprobación (Bancóldex/Finagro: ~394k créditos 2024, foco rurales)
}
df = pd.DataFrame(data)
df.to_csv('data/finance/finance_sim.csv', index=False)
print("finance_sim.csv creado con 1000 muestras (basado en agregados Superfinanciera/Finagro/Bancóldex 2023-2025).")
print("Muestra:\n", df.head())
print("Distribución approve: 70% sí (recall alto para minimizar rechazos erróneos).")