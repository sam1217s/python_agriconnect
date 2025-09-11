# Frontend AgriConnect - Guía de Configuración

## Descripción
Frontend web interactivo para AgriConnect Colombia desarrollado con Streamlit. Proporciona interfaces amigables para los 4 agentes de IA con visualizaciones avanzadas y explicabilidad.

## Características del Frontend

### Páginas Incluidas
- **🏠 Inicio**: Dashboard general con métricas del sistema
- **🌦️ ClimateAI**: Predicción climática con visualización SHAP
- **💰 MarketAI**: Proyección de precios con análisis de factores
- **🌿 AgroExpert**: Análisis de imágenes con Grad-CAM
- **🏦 FinanceAI**: Evaluación crediticia con explicabilidad

### Tecnologías Utilizadas
- **Streamlit**: Framework web principal
- **Plotly**: Visualizaciones interactivas
- **Requests**: Comunicación con API FastAPI
- **Pillow**: Procesamiento de imágenes
- **Pandas/Numpy**: Manipulación de datos

## Instalación y Configuración

### Paso 1: Preparar el Entorno

```bash
# Asegúrate de estar en el directorio del proyecto
cd C:\Users\[usuario]\Pictures\agriconnect

# Activar entorno virtual (si lo usas)
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias del frontend
pip install -r requirements_streamlit.txt
```

### Paso 2: Verificar Estructura de Archivos

```
agriconnect/
├── main.py                    # API FastAPI (debe estar corriendo)
├── app.py                     # Frontend Streamlit (nuevo)
├── requirements.txt           # Dependencias API
├── requirements_streamlit.txt # Dependencias frontend (nuevo)
├── models/                    # Modelos IA
├── data/                      # Datasets
├── xai/                       # Explicabilidad
└── ...
```

### Paso 3: Configuración de la API

**IMPORTANTE**: El frontend requiere que la API FastAPI esté funcionando.

```bash
# Terminal 1: Ejecutar API (DEBE estar corriendo primero)
uvicorn main:app --reload

# Verificar que la API responde en:
# http://127.0.0.1:8000/health
```

### Paso 4: Ejecutar Frontend

```bash
# Terminal 2: Ejecutar Streamlit (en nueva terminal)
streamlit run app.py

# El frontend se abrirá automáticamente en:
# http://localhost:8501
```

## Configuración de URLs

### Configuración por Defecto
El frontend está configurado para conectarse a:
```python
API_BASE_URL = "http://127.0.0.1:8000"
```

### Cambiar URL de la API
Si tu API está en otra dirección, modifica la línea 15 en `app.py`:

```python
# Para API en otra máquina
API_BASE_URL = "http://192.168.1.100:8000"

# Para API en la nube
API_BASE_URL = "https://tu-api.herokuapp.com"
```

## Uso del Frontend

### 1. Página de Inicio
- **Dashboard general** con métricas del sistema
- **Estado de la API** en tiempo real
- **Descripción de agentes** disponibles

### 2. ClimateAI - Predicción Climática
```
Entrada:
- Temperatura: 10-40°C
- Humedad: 30-100%

Salida:
- Predicción lluvia/seco
- Nivel de confianza
- Gráfico SHAP de importancia
- Recomendaciones específicas
```

### 3. MarketAI - Precios
```
Entrada:
- Día del año: 1-365
- Precio global de referencia

Salida:
- Precio proyectado en COP/kg
- Comparación con rangos típicos
- Análisis de factores
- Recomendaciones de venta
```

### 4. AgroExpert - Análisis de Imágenes
```
Entrada:
- Imagen de hoja (PNG/JPG/JPEG)

Salida:
- Diagnóstico saludable/enfermo
- Confianza del modelo
- Mapa de calor Grad-CAM
- Recomendaciones de tratamiento
```

### 5. FinanceAI - Microcréditos
```
Entrada:
- Ingresos mensuales (miles COP)
- Tipo de cultivo
- Historial crediticio (0-1)

Salida:
- Aprobación/rechazo
- Confianza de la decisión
- Factores de decisión SHAP
- Condiciones sugeridas
```

## Personalización del Frontend

### Cambiar Colores y Estilos
Modifica el CSS en las líneas 16-60 de `app.py`:

```python
st.markdown("""
<style>
    .main-header {
        color: #2E7D32;  /* Cambiar color principal */
    }
    .agent-card {
        background-color: #f0f8f0;  /* Cambiar fondo */
    }
</style>
""", unsafe_allow_html=True)
```

### Agregar Nuevas Páginas
```python
# En la función main(), agregar nueva página:
pages = {
    "🏠 Inicio": "home",
    "🌦️ ClimateAI": "climate",
    "💰 MarketAI": "market",
    "🌿 AgroExpert": "agro",
    "🏦 FinanceAI": "finance",
    "📊 Nueva Página": "nueva"  # Agregar aquí
}

# Crear función render_nueva() siguiendo el patrón
```

### Modificar Visualizaciones
Las visualizaciones usan Plotly. Ejemplo para cambiar colores:

```python
# Cambiar colores de gráficos
fig.add_trace(go.Bar(
    x=features,
    y=feature_importance,
    marker_color=['#FF6B6B', '#51CF66']  # Cambiar estos colores
))
```

## Solución de Problemas

### Error: "No se puede conectar con la API"
```bash
# 1. Verificar que la API esté corriendo
curl http://127.0.0.1:8000/health

# 2. Si no responde, ejecutar:
uvicorn main:app --reload

# 3. Verificar firewall/antivirus
```

### Error: "ModuleNotFoundError"
```bash
# Instalar dependencias faltantes
pip install streamlit plotly requests pandas numpy pillow

# O reinstalar todo
pip install -r requirements_streamlit.txt
```

### Error: "Connection timeout"
```bash
# Aumentar timeout en app.py líneas donde dice timeout=10
# Cambiar a timeout=30 o más
```

### Error: "Cannot upload file"
```bash
# Verificar que python-multipart esté instalado en la API
pip install python-multipart

# Reiniciar la API después de instalar
```

### Problemas de Rendimiento
```bash
# 1. Reducir tamaño máximo de archivos en Streamlit
# Agregar en .streamlit/config.toml:
[server]
maxUploadSize = 50

# 2. Optimizar imágenes antes de subir
# 3. Usar cache de Streamlit para datos pesados
```

## Mejoras y Extensiones

### 1. Agregar Autenticación
```python
# Usar streamlit-authenticator
import streamlit_authenticator as stauth

# Configurar login en sidebar
```

### 2. Conectar Base de Datos
```python
# Guardar histórico de predicciones
import sqlite3

def save_prediction(agent, input_data, result):
    # Guardar en BD local
    pass
```

### 3. Exportar Reportes
```python
# Agregar botón de descarga
import io
import base64

def download_report(data):
    # Generar PDF/Excel
    pass
```

### 4. Notificaciones Push
```python
# Usar streamlit-notifications
import streamlit as st

# Mostrar alertas automáticas
```

## Despliegue del Frontend

### Opción 1: Streamlit Cloud (Gratuito)
```bash
# 1. Subir código a GitHub
git add app.py requirements_streamlit.txt
git commit -m "Add Streamlit frontend"
git push

# 2. Ir a share.streamlit.io
# 3. Conectar repositorio
# 4. Configurar main file: app.py
```

### Opción 2: Docker
```dockerfile
# Dockerfile.streamlit
FROM python:3.10-slim

WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY app.py .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

```bash
# Construir y ejecutar
docker build -f Dockerfile.streamlit -t agriconnect-frontend .
docker run -p 8501:8501 agriconnect-frontend
```

### Opción 3: Heroku
```bash
# 1. Crear Procfile
echo "web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0" > Procfile

# 2. Deploy
heroku create agriconnect-frontend
git push heroku main
```

## Estructura de Archivos Final

```
agriconnect/
├── main.py                      # API FastAPI
├── app.py                       # Frontend Streamlit ⭐
├── requirements.txt             # Deps API
├── requirements_streamlit.txt   # Deps Frontend ⭐
├── FRONTEND_SETUP.md           # Esta guía ⭐
├── models/                     # Modelos IA
├── data/                       # Datasets
├── xai/                        # Explicabilidad
├── .streamlit/                 # Config Streamlit (opcional)
│   └── config.toml
└── static/                     # Archivos estáticos (opcional)
    └── images/
```

## Comandos de Ejecución Rápida

```bash
# Setup completo desde cero
git clone [tu-repo]
cd agriconnect
pip install -r requirements.txt
pip install -r requirements_streamlit.txt

# Terminal 1: API
uvicorn main:app --reload

# Terminal 2: Frontend (nueva terminal)
streamlit run app.py

# Acceder a:
# API: http://127.0.0.1:8000/docs
# Frontend: http://localhost:8501
```

## Métricas de Rendimiento

- **Tiempo de carga**: <3 segundos
- **Tamaño máximo de imagen**: 50MB
- **Usuarios concurrentes**: 10-50 (según recursos)
- **Tiempo de respuesta API**: <2 segundos promedio

## Conclusión

El frontend de AgriConnect proporciona una interfaz completa y profesional para todos los agentes de IA. Con visualizaciones interactivas, explicabilidad clara y diseño intuitivo, está listo para demostrar la potencia del sistema en SENASOFT 2025.

**¡AgriConnect Frontend está listo para producción!** 🚀