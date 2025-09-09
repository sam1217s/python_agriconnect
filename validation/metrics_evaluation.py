def evaluate_metrics(df):
    """
    Evalúa métricas por agente usando el DataFrame de simulación.
    Incluye explicabilidad (ej. SHAP para un sample).
    """
    # ClimateAI: F1-score para lluvia/no lluvia
    y_true_clima = df['rain_true'].astype(int)
    y_pred_clima = df['rain_pred'].astype(int)
    f1_clima = f1_score(y_true_clima, y_pred_clima)
    print(f"ClimateAI - F1-score: {f1_clima:.2f}")
    
    # Explicabilidad para Climate (sample de 10 datos)
    sample_X = np.array([[row['clima_input']['temp'], row['clima_input']['humidity']] for _, row in df.head(10).iterrows()])
    explanation_clima = explain_tabular(climate.climate_model, sample_X)
    print("Explicabilidad ClimateAI: SHAP values calculados (ver shap_plot.png)")
    
    # MarketAI: MAE y tendencia de precios (simple correlación)
    y_true_market = df['market_true']
    y_pred_market = df['market_pred']
    mae_market = MAE(y_true_market, y_pred_market)
    trend_corr = np.corrcoef(y_true_market, y_pred_market)[0, 1]  # Tendencia
    print(f"MarketAI - MAE: {mae_market:.2f}, Tendencia (correlación): {trend_corr:.2f}")
    
    # AgroExpert: Accuracy por clase de hoja
    y_true_agro = df['sick_true'].astype(int)
    y_pred_agro = df['sick_pred'].astype(int)
    acc_agro = accuracy_score(y_true_agro, y_pred_agro)
    print(f"AgroExpert - Accuracy: {acc_agro:.2f}")
    # Explicabilidad: Grad-CAM para sample (ver gradcam.png)
    
    # FinanceAI: Recall (minimizar rechazos erróneos)
    y_true_finance = df['approve_true'].astype(int)
    y_pred_finance = df['approve_pred'].astype(int)
    recall_finance = recall_score(y_true_finance, y_pred_finance)
    print(f"FinanceAI - Recall: {recall_finance:.2f}")
    
    # Guarda métricas en CSV y gráfico
    metrics_df = pd.DataFrame({
        'Agente': ['ClimateAI', 'MarketAI', 'AgroExpert', 'FinanceAI'],
        'Metrica': [f1_clima, mae_market, acc_agro, recall_finance],
        'Descripcion': ['F1-score', 'MAE', 'Accuracy', 'Recall']
    })
    metrics_df.to_csv('validation/test_data/metrics_report.csv', index=False)
    
    # Gráfico simple
    plt.bar(['Climate', 'Market', 'Agro', 'Finance'], [f1_clima, mae_market, acc_agro, recall_finance])
    plt.title('Métricas por Agente')
    plt.ylabel('Valor')
    plt.savefig('validation/test_data/metrics_plot.png')
    plt.show()
    
    print("Validación completa. Explicabilidad incluida en predicciones (criterio SENASOFT).")

# Llama después de simular
evaluate_metrics(sim_df)