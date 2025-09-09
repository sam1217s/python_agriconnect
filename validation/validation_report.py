from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def generate_report():
    """
    Genera reporte PDF con métricas, explicabilidad y ética.
    """
    c = canvas.Canvas("validation/validation_report.pdf", pagesize=letter)
    c.drawString(100, 750, "Reporte de Validación - AgriConnect Colombia")
    c.drawString(100, 730, "Entorno simulado con 15 usuarios virtuales.")
    c.drawString(100, 710, "Métricas: Ver CSV y plots generados.")
    c.drawString(100, 690, "Explicabilidad: SHAP y Grad-CAM aplicados (imágenes guardadas).")
    c.drawString(100, 670, "Ética: No se almacenan datos sensibles; validación humana recomendada.")
    c.drawString(100, 650, "Impacto: Mejora seguridad alimentaria, alineado con CONPES 4144.")
    c.save()
    print("Reporte PDF generado: validation_report.pdf")

generate_report()