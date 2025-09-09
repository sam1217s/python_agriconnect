import shap
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def explain_tabular(model, X):
    """
    Genera explicaciones SHAP sin matplotlib (thread-safe)
    """
    try:
        # Detectar tipo de modelo
        model_type = str(type(model)).lower()
        
        if 'randomforest' in model_type or 'tree' in model_type:
            # Para modelos basados en árboles
            explainer = shap.TreeExplainer(model)
        else:
            # Para modelos lineales y otros
            explainer = shap.LinearExplainer(model, X)
        
        # Calcular valores SHAP
        shap_values = explainer.shap_values(X)
        
        # Si es clasificación binaria, tomar la clase positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        # Retornar valores SHAP como array numpy
        return np.array(shap_values)
        
    except Exception as e:
        # Fallback: retornar explicación simple basada en coeficientes
        print(f"SHAP fallback: {e}")
        return explain_fallback(model, X)

def explain_fallback(model, X):
    """
    Explicación fallback cuando SHAP falla
    """
    try:
        # Para modelos lineales, usar coeficientes
        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            explanation = X * coef
            return explanation
        
        # Para otros modelos, usar importancia de features
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            explanation = X * importance
            return explanation
        
        # Fallback general: valores uniformes
        else:
            return np.ones_like(X) * 0.1
            
    except Exception as e:
        print(f"Fallback failed: {e}")
        # Último recurso: array de ceros
        return np.zeros_like(X)