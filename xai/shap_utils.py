import shap
import matplotlib.pyplot as plt

def explain_tabular(model, X):
    explainer = shap.TreeExplainer(model) if 'tree' in str(type(model)) else shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    plt.savefig('xai/shap_plot.png')  # Para demo
    return shap_values