# 🌱 AgriConnect Colombia - IA para la Agricultura Rural

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0+-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![SENASOFT](https://img.shields.io/badge/SENASOFT-2025-red.svg)](https://www.sena.edu.co/)

**"El campo sabe su clima. AgriConnect lo conecta: donde el conocimiento campesino potencia la inteligencia artificial."**

[🚀 Demo en Vivo](http://127.0.0.1:8000/docs) • [📊 Documentación API](http://127.0.0.1:8000/docs) • [🐛 Reportar Bug](https://github.com/sam1217s/python_agriconnect/issues)

</div>

---

## 📝 Descripción

**AgriConnect** es una solución de inteligencia artificial explicable diseñada para empoderar a pequeños productores rurales en Colombia. Integra cuatro agentes IA especializados que trabajan en conjunto para optimizar la producción agrícola y reducir las brechas digitales en el campo.

### 🤖 Agentes IA Disponibles:

| Agente | Función | Tecnología | Métrica Actual |
|--------|---------|------------|----------------|
| **🌦️ ClimateAI** | Predicción climática lluvia/seco | LogisticRegression + SHAP | F1-score: 0.41 |
| **💰 MarketAI** | Proyección de precios agrícolas | LinearRegression + SHAP | MAE: 651.99 COP/kg |
| **🌿 AgroExpert** | Diagnóstico de enfermedades en hojas | MobileNetV2 + Grad-CAM | Accuracy: 0.73 |
| **🏦 FinanceAI** | Recomendación de microcréditos | RandomForest + SHAP | Recall: 0.80 |

---

## 🏗️ Arquitectura del Sistema

```
AgriConnect/
├── 🚀 main.py                 # API FastAPI central
├── 🤖 models/                 # Modelos de IA
│   ├── climate.py            # ClimateAI (predicción climática)
│   ├── market.py             # MarketAI (precios)
│   ├── agro.py               # AgroExpert (visión por computadora)
│   └── finance.py            # FinanceAI (microcréditos)
├── 📊 data/                   # Datasets y simulaciones
│   ├── weather/              # Datos climáticos IDEAM
│   ├── prices/               # Precios SIPSA/DANE
│   ├── finance/              # Datos financieros simulados
│   └── leaves/               # ~54k imágenes PlantVillage
├── 🔍 xai/                    # Explicabilidad (SHAP + Grad-CAM)
├── ✅ validation/             # Simulación usuarios y métricas
└── 🐳 Dockerfile             # Contenedor para despliegue
```

---

## ⚡ Instalación Rápida

### Prerrequisitos
- Python 3.10+
- Git
- 4GB RAM mínimo (para TensorFlow)

### 🔧 Configuración

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/sam1217s/python_agriconnect.git
   cd python_agriconnect
   ```

2. **Crea entorno virtual (recomendado):**
   ```bash
   # Con conda
   conda create -n agriconnect python=3.10
   conda activate agriconnect
   
   # O con virtualenv
   python -m venv agriconnect
   source agriconnect/bin/activate  # Linux/Mac
   # o
   agriconnect\Scripts\activate     # Windows
   ```

3. **Instala dependencias:**
   ```bash
   pip install -r requirements.txt
   pip install python-multipart  # Para subida de archivos
   ```

4. **Lanza la API:**
   ```bash
   uvicorn main:app --reload
   ```

5. **¡Listo!** Ve a http://127.0.0.1:8000/docs

---

## 🎯 Uso de la API

### 📡 Endpoints Disponibles

#### 🌦️ **ClimateAI** - Predicción Climática
```bash
curl -X POST "http://127.0.0.1:8000/clima" \
  -H "Content-Type: application/json" \
  -d '{"temp": 25.5, "humidity": 75.0}'
```
**Respuesta:**
```json
{
  "rain": false,
  "explanation": [[0.12, -0.05]],
  "confidence": 0.76,
  "features": ["temperature", "humidity"]
}
```

#### 💰 **MarketAI** - Proyección de Precios
```bash
curl -X POST "http://127.0.0.1:8000/mercado" \
  -H "Content-Type: application/json" \
  -d '{"day": 250, "global_price": 350.0}'
```
**Respuesta:**
```json
{
  "price": 3289.29,
  "currency": "COP/kg",
  "explanation": [[125.0, 2164.29]],
  "features": ["day_of_year", "global_price"]
}
```

#### 🌿 **AgroExpert** - Diagnóstico de Hojas
```bash
curl -X POST "http://127.0.0.1:8000/agro" \
  -F "file=@hoja_cafe.jpg"
```
**Respuesta:**
```json
{
  "sick": true,
  "confidence": 0.89,
  "explanation_heatmap": [...],
  "filename": "hoja_cafe.jpg"
}
```

#### 🏦 **FinanceAI** - Evaluación de Microcrédito
```bash
curl -X POST "http://127.0.0.1:8000/finanzas" \
  -H "Content-Type: application/json" \
  -d '{"income": 3500.0, "crop_type_encoded": 2, "credit_history": 0.75}'
```
**Respuesta:**
```json
{
  "approve": true,
  "confidence": 0.89,
  "explanation": [[-0.12, 0.12], [-0.02, 0.02], [-0.03, 0.03]],
  "features": ["income", "crop_type", "credit_history"]
}
```

---

## 📊 Fuentes de Datos y Validación

### 🌡️ **Datos Climáticos**
- **Fuente**: IDEAM (Instituto de Hidrología, Meteorología y Estudios Ambientales)
- **Cobertura**: 2023-2025, proyecciones transición La Niña
- **Variables**: Temperatura (20-28°C), Humedad (60-85%), Precipitación (30%)
- **Dataset**: [datos.gov.co ID sbwg-7ju4](https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/Temperatura-Ambiente-del-Aire/sbwg-7ju4)

### 💹 **Precios de Mercado**
- **Fuente**: SIPSA/DANE (Sistema de Información de Precios del Sector Agropecuario)
- **Productos**: Café (3,500 COP/kg), Cacao (4,000), Aguacate (2,500)
- **Frecuencia**: Semanal (Semana 35, 2025)
- **Dataset**: [Boletín SIPSA](https://www.dane.gov.co/index.php/estadisticas-por-tema/agropecuario/sistema-de-informacion-de-precios-sipsa)

### 🏦 **Datos Financieros**
- **Fuente**: Finagro/Superfinanciera (394k créditos 2024)
- **Perfil**: 90% rurales/pequeños, $7-8M COP promedio
- **Simulación**: 1,000 registros éticos (ingresos, historial, cultivo)
- **Dataset**: [datos.gov.co](https://www.datos.gov.co/Agricultura-y-Desarrollo-Rural/Colocaciones-de-Cr-dito-Sector-Agropecuario-2021-2/w3uf-w9ey)

### 🌿 **Imágenes de Cultivos**
- **Fuente**: PlantVillage Dataset (Kaggle)
- **Volumen**: ~54,000 imágenes, 38 clases
- **Distribución**: 5,262 hojas sanas / 13,148 enfermas
- **Dataset**: [PlantVillage Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## 🎯 Métricas de Rendimiento

| Modelo | Métrica | Valor Actual | Meta | Estado |
|--------|---------|--------------|------|--------|
| ClimateAI | F1-score | 0.41 | 0.70+ | 🟡 En mejora |
| MarketAI | MAE | 651.99 COP/kg | <200 | 🟡 En mejora |
| AgroExpert | Accuracy | 0.73 | 0.85+ | 🟢 Bueno |
| FinanceAI | Recall | 0.80 | 0.80+ | 🟢 Excelente |

---

## 🚀 Despliegue

### 🐳 Docker (Recomendado)
```bash
# Construir imagen
docker build -t agriconnect .

# Ejecutar contenedor
docker run -p 8000:80 agriconnect
```

### ☁️ Google Cloud Run
```bash
gcloud run deploy agriconnect \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 🔧 Producción Local
```bash
# Con Gunicorn
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

# Con PM2 (Node.js ecosystem)
npm install -g pm2
pm2 start "uvicorn main:app --host 0.0.0.0 --port 8000" --name agriconnect
```

---

## 🛠️ Solución de Problemas

### ❌ Error: "set_wakeup_fd only works in main thread"
```bash
# Actualiza xai/shap_utils.py y reinicia
uvicorn main:app --reload
```

### ❌ Error: "python-multipart" requerido
```bash
pip install python-multipart
```

### ❌ TensorFlow no encuentra CUDA
```bash
# CPU-only (funcional)
pip install tensorflow-cpu

# GPU (opcional, mejor rendimiento)
pip install tensorflow[and-cuda]
```

### ❌ Memoria insuficiente
```bash
# Reducir batch size en models/agro.py
batch_size=16  # en lugar de 32
```

---

## 🧪 Testing y Validación

### 🔍 Tests Automatizados
```bash
# Instalar dependencias de testing
pip install pytest httpx

# Ejecutar tests
pytest tests/ -v

# Coverage
pytest --cov=models tests/
```

### 👥 Simulación de Usuarios
```bash
# Generar 15 usuarios virtuales con 3 reportes cada uno
python validation/simulate_users.py

# Evaluar métricas
python validation/metrics_evaluation.py
```

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Sigue estos pasos:

1. **Fork** del repositorio
2. **Crea** una rama: `git checkout -b feature/nueva-funcionalidad`
3. **Commit** tus cambios: `git commit -m 'Add: nueva funcionalidad'`
4. **Push** a la rama: `git push origin feature/nueva-funcionalidad`
5. **Abre** un Pull Request

### 🐛 Reportar Bugs
- Usa [GitHub Issues](https://github.com/sam1217s/python_agriconnect/issues)
- Incluye logs completos y pasos para reproducir
- Especifica versión de Python y OS

---

## 🌍 Impacto Social

### 📈 **Objetivos de Desarrollo Sostenible (ODS)**
- 🎯 **ODS 2**: Hambre Cero - Mejora seguridad alimentaria
- 🎯 **ODS 8**: Trabajo Decente - Empodera productores rurales  
- 🎯 **ODS 9**: Industria e Innovación - Reduce brechas digitales
- 🎯 **ODS 13**: Acción Climática - Adaptación al cambio climático

### 🏛️ **Marco Regulatorio**
- ✅ **CONPES 4144 (2021)**: Estrategia Nacional de IA
- ✅ **Ley 1266/2008**: Protección datos personales (datos anonimizados)
- ✅ **Plan Nacional TIC**: Conectividad rural
- ✅ **Agenda 2030**: Sostenibilidad y equidad

---

## 📞 Contacto y Soporte

<div align="center">

**Desarrollado con ❤️ por sam1217s para SENASOFT 2025**

[![GitHub](https://img.shields.io/badge/GitHub-sam1217s-black?logo=github)](https://github.com/sam1217s)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-blue?logo=linkedin)](https://linkedin.com/in/sam1217s)
[![Email](https://img.shields.io/badge/Email-Contacto-red?logo=gmail)](mailto:sam1217s@example.com)

**AgriConnect Colombia** - Conectando el campo con la inteligencia artificial 🇨🇴

</div>

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

```
MIT License - Copyright (c) 2025 sam1217s

Se permite el uso comercial, modificación, distribución y uso privado.
Incluir aviso de copyright y licencia en todas las copias.
```

---

<div align="center">

**⭐ Si AgriConnect te parece útil, ¡dale una estrella en GitHub! ⭐**

*Hecho con Python 🐍, FastAPI ⚡, TensorFlow 🧠 y mucho ☕ colombiano*

</div>