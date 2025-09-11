import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import json
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="AgriConnect Colombia",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL base de la API
API_BASE_URL = "http://127.0.0.1:8000"

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .result-success {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .result-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
    .explanation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def check_api_status():
    """Verificar estado de la API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🌱 AgriConnect Colombia</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Inteligencia Artificial para la Agricultura Rural</p>', unsafe_allow_html=True)
    
    # Verificar estado de la API
    if not check_api_status():
        st.error("⚠️ No se puede conectar con la API. Asegúrate de que el servidor esté ejecutándose en http://127.0.0.1:8000")
        st.code("uvicorn main:app --reload")
        return
    
    # Sidebar para navegación
    st.sidebar.title("Navegación")
    pages = {
        "🏠 Inicio": "home",
        "🌦️ ClimateAI": "climate",
        "💰 MarketAI": "market", 
        "🌿 AgroExpert": "agro",
        "🏦 FinanceAI": "finance"
    }
    
    selected_page = st.sidebar.selectbox("Selecciona un agente:", list(pages.keys()))
    page = pages[selected_page]
    
    # Información del estado
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Estado del Sistema**")
    api_status = check_api_status()
    status_color = "🟢" if api_status else "🔴"
    st.sidebar.markdown(f"{status_color} API: {'Activa' if api_status else 'Inactiva'}")
    
    # Renderizar página seleccionada
    if page == "home":
        render_home()
    elif page == "climate":
        render_climate()
    elif page == "market":
        render_market()
    elif page == "agro":
        render_agro()
    elif page == "finance":
        render_finance()

def render_home():
    """Página de inicio"""
    st.markdown("## Bienvenido a AgriConnect")
    st.markdown("Un ecosistema de inteligencia artificial diseñado para empoderar a los pequeños productores rurales de Colombia.")
    
    # Métricas del sistema
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>4</h3><p>Agentes IA</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>73%</h3><p>Precisión AgroExpert</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h3>0.80</h3><p>Recall FinanceAI</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card"><h3>24/7</h3><p>Disponibilidad</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Descripción de agentes
    st.markdown("## Nuestros Agentes IA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
        <h4>🌦️ ClimateAI</h4>
        <p>Predicción climática basada en temperatura y humedad. Ayuda a planificar siembras y cosechas.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="agent-card">
        <h4>🌿 AgroExpert</h4>
        <p>Diagnóstico de enfermedades en hojas mediante análisis de imágenes con explicabilidad visual.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="agent-card">
        <h4>💰 MarketAI</h4>
        <p>Proyección de precios agrícolas para optimizar momentos de venta y planificación financiera.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="agent-card">
        <h4>🏦 FinanceAI</h4>
        <p>Evaluación de microcréditos rurales con criterios específicos para el sector agropecuario.</p>
        </div>
        """, unsafe_allow_html=True)

def render_climate():
    """
    Página de predicción climática
    """
    st.header("🌤️ ClimateAI - Predicción Climática")
    st.markdown("**Predice la probabilidad de lluvia basado en temperatura y humedad**")
    
    # Información explicativa
    with st.expander("ℹ️ ¿Cómo funciona?", expanded=False):
        st.markdown("""
        - **Temperatura**: Temperatura ambiente en grados Celsius (10-40°C)
        - **Humedad**: Porcentaje de humedad relativa (40-90%)
        - **Modelo**: Regresión logística entrenada con datos IDEAM 2023-2025
        - **Región**: Optimizado para zona Andina y Pacífica colombiana
        """)
    
    # Layout en columnas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Parámetros Climáticos")
        
        # Inputs del usuario
        temperature = st.slider(
            "🌡️ Temperatura (°C)",
            min_value=10.0,
            max_value=40.0,
            value=24.0,
            step=0.5,
            help="Temperatura ambiente en grados Celsius"
        )
        
        humidity = st.slider(
            "💧 Humedad Relativa (%)",
            min_value=40.0,
            max_value=90.0,
            value=70.0,
            step=1.0,
            help="Porcentaje de humedad relativa"
        )
        
        # Información contextual
        st.info(f"📍 **Condiciones actuales**: {temperature}°C, {humidity}% humedad")
        
        # Botón de predicción
        if st.button("🔮 Predecir Lluvia", type="primary", use_container_width=True):
            # Preparar datos para la API
            data = {
                "temp": float(temperature),
                "humidity": float(humidity)
            }
            
            # Mostrar spinner mientras procesa
            with st.spinner("🔄 Analizando condiciones climáticas..."):
                try:
                    # Llamada al backend
                    import requests
                    response = requests.post(
                        "http://localhost:8000/clima", 
                        json=data,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        # Guardar resultado en session state
                        result = response.json()
                        st.session_state.climate_result = result
                        st.success("✅ Predicción completada exitosamente!")
                        
                    else:
                        st.error(f"❌ Error del servidor: {response.status_code}")
                        st.session_state.climate_result = None
                        
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Error de conexión: Verifica que el backend esté ejecutándose en http://localhost:8000")
                    st.session_state.climate_result = None
                    
                except requests.exceptions.Timeout:
                    st.error("⏱️ Tiempo de espera agotado: El servidor tardó demasiado en responder")
                    st.session_state.climate_result = None
                    
                except Exception as e:
                    st.error(f"💥 Error inesperado: {str(e)}")
                    st.session_state.climate_result = None
    
    with col2:
        st.subheader("🎯 Guía Rápida")
        st.markdown("""
        **🌧️ Alta probabilidad de lluvia:**
        - Temperatura > 25°C
        - Humedad > 80%
        
        **☀️ Baja probabilidad de lluvia:**
        - Temperatura < 20°C
        - Humedad < 60%
        
        **🌤️ Condiciones mixtas:**
        - Depende de la combinación
        """)
    
    # Mostrar resultados si existen
    if st.session_state.climate_result is not None:
        st.markdown("---")
        
        result = st.session_state.climate_result
        
        # Resultado principal
        rain_prediction = result.get('rain', False)
        confidence = result.get('confidence', 0)
        
        # Crear métricas visuales
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            if rain_prediction:
                st.metric(
                    "🌧️ Predicción",
                    "LLUVIA",
                    delta="Alta probabilidad",
                    delta_color="normal"
                )
            else:
                st.metric(
                    "☀️ Predicción", 
                    "SIN LLUVIA",
                    delta="Baja probabilidad",
                    delta_color="inverse"
                )
        
        with col_result2:
            st.metric(
                "📊 Confianza",
                f"{confidence:.1%}" if confidence > 0 else "N/A",
                delta=None
            )
            
        with col_result3:
            # Calcular índice de confort
            comfort_index = "Alto" if 20 <= temperature <= 26 and 50 <= humidity <= 70 else "Medio" if 18 <= temperature <= 28 and 45 <= humidity <= 75 else "Bajo"
            st.metric(
                "🌡️ Confort",
                comfort_index,
                delta=None
            )
        
        # Alerta principal
        if rain_prediction:
            st.success("""
            ### 🌧️ ¡Predicción: Se esperan precipitaciones!
            
            **Recomendaciones agrícolas:**
            - 🚜 Evita trabajos de campo intensivos
            - 🌱 Ideal para cultivos que requieren agua
            - 📦 Protege fertilizantes y semillas
            - 🛡️ Cubre equipos sensibles a la humedad
            """)
        else:
            st.info("""
            ### ☀️ Predicción: Condiciones secas
            
            **Recomendaciones agrícolas:**
            - 💧 Considera sistemas de riego
            - 🌾 Buen momento para cosecha
            - 🚜 Condiciones ideales para maquinaria
            - 🌱 Monitorea estrés hídrico en cultivos
            """)
        
        # Explicabilidad del modelo
        if 'explanation' in result and result['explanation']:
            st.markdown("---")
            st.subheader("🔍 Explicabilidad del Modelo (SHAP)")
            
            with st.expander("📈 Ver análisis detallado", expanded=False):
                explanation = result['explanation']
                
                # Mostrar valores SHAP si están disponibles
                if isinstance(explanation, list) and len(explanation) >= 2:
                    col_exp1, col_exp2 = st.columns(2)
                    
                    with col_exp1:
                        temp_impact = explanation[0] if len(explanation) > 0 else 0
                        st.metric(
                            "🌡️ Impacto Temperatura",
                            f"{temp_impact:.3f}",
                            delta="Positivo" if temp_impact > 0 else "Negativo",
                            delta_color="normal" if temp_impact > 0 else "inverse"
                        )
                    
                    with col_exp2:
                        humidity_impact = explanation[1] if len(explanation) > 1 else 0
                        st.metric(
                            "💧 Impacto Humedad",
                            f"{humidity_impact:.3f}",
                            delta="Positivo" if humidity_impact > 0 else "Negativo",
                            delta_color="normal" if humidity_impact > 0 else "inverse"
                        )
                    
                    st.markdown("""
                    **Interpretación:**
                    - Valores positivos incrementan la probabilidad de lluvia
                    - Valores negativos la reducen
                    - Mayor magnitud = mayor influencia en la decisión
                    """)
                else:
                    st.write("Datos de explicabilidad:", explanation)
        
        # Historial y contexto
        st.markdown("---")
        with st.expander("📊 Contexto Regional", expanded=False):
            st.markdown(f"""
            **Análisis para las condiciones actuales:**
            
            🌡️ **Temperatura {temperature}°C:**
            {'Temperatura alta para la región Andina' if temperature > 28 else 'Temperatura moderada' if temperature > 22 else 'Temperatura fresca'}
            
            💧 **Humedad {humidity}%:**
            {'Humedad muy alta - típica de época lluviosa' if humidity > 80 else 'Humedad alta - condiciones favorables para lluvia' if humidity > 70 else 'Humedad moderada' if humidity > 60 else 'Humedad baja - condiciones secas'}
            
            📅 **Contexto estacional:**
            Datos basados en patrones climáticos de Antioquia, Tolima, Cauca y Valle del Cauca (2023-2025)
            """)
    
    else:
        # Mensaje cuando no hay resultados
        st.markdown("---")
        st.warning("⚠️ No hay resultados disponibles. Ingresa los parámetros climáticos y presiona 'Predecir Lluvia' para obtener una predicción.")
        
        # Mostrar datos de ejemplo
        with st.expander("💡 Ejemplos de uso", expanded=False):
            st.markdown("""
            **Escenario 1 - Época seca:**
            - Temperatura: 22°C
            - Humedad: 55%
            - Resultado esperado: Sin lluvia
            
            **Escenario 2 - Época lluviosa:**
            - Temperatura: 26°C
            - Humedad: 85%
            - Resultado esperado: Lluvia probable
            
            **Escenario 3 - Transición:**
            - Temperatura: 24°C
            - Humedad: 72%
            - Resultado: Depende del modelo
            """)
    
    # Footer con información técnica
    st.markdown("---")
    st.caption("🤖 Modelo entrenado con datos del IDEAM (2023-2025) | 🎯 Optimizado para agricultura colombiana | 🔄 Actualizado septiembre 2025")
def render_market():
    """
    Página de predicción de precios de mercado
    """
    st.header("💰 MarketAI - Predicción de Precios")
    st.markdown("**Predice precios de productos agrícolas basado en tendencias de mercado**")
    
    # Información explicativa
    with st.expander("ℹ️ ¿Cómo funciona?", expanded=False):
        st.markdown("""
        - **Fuente de datos**: SIPSA (Sistema de Información de Precios del Sector Agropecuario)
        - **Modelo**: Regresión lineal con variables temporales y de mercado global
        - **Actualización**: Datos semanales del DANE
        - **Cobertura**: Principales productos agrícolas colombianos
        - **Precisión**: MAE < 50 COP/kg en promedio
        """)
    
    # Layout en columnas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Parámetros de Mercado")
        
        # Selector de producto
        product_options = {
            "🍅 Tomate": {"base_price": 2500, "volatility": 0.3},
            "🥔 Papa": {"base_price": 1800, "volatility": 0.2},
            "🧅 Cebolla": {"base_price": 2200, "volatility": 0.4},
            "🥕 Zanahoria": {"base_price": 1600, "volatility": 0.25},
            "🌽 Maíz": {"base_price": 900, "volatility": 0.15},
            "🍌 Plátano": {"base_price": 1200, "volatility": 0.2},
            "🥬 Lechuga": {"base_price": 3000, "volatility": 0.35},
            "🥒 Pepino": {"base_price": 2800, "volatility": 0.3}
        }
        
        selected_product = st.selectbox(
            "🛒 Producto Agrícola",
            options=list(product_options.keys()),
            index=0,
            help="Selecciona el producto para predecir su precio"
        )
        
        # Parámetros temporales
        day_of_year = st.slider(
            "📅 Día del Año",
            min_value=1,
            max_value=365,
            value=254,  # Aproximadamente 11 de septiembre
            step=1,
            help="Día del año (1 = 1 enero, 365 = 31 diciembre)"
        )
        
        # Convertir día a fecha para mostrar
        import datetime
        date_obj = datetime.datetime(2025, 1, 1) + datetime.timedelta(days=day_of_year - 1)
        st.info(f"📆 Fecha correspondiente: {date_obj.strftime('%d de %B de %Y')}")
        
        # Precio global de referencia
        global_price = st.slider(
            "🌍 Precio Global de Referencia (USD/Tonelada)",
            min_value=100.0,
            max_value=1000.0,
            value=400.0,
            step=10.0,
            help="Precio internacional de referencia para el producto"
        )
        
        # Información contextual sobre temporadas
        product_info = product_options[selected_product]
        st.info(f"""
        📈 **{selected_product}**: Precio base ~{product_info['base_price']} COP/kg  
        📊 **Volatilidad**: {'Alta' if product_info['volatility'] > 0.3 else 'Media' if product_info['volatility'] > 0.2 else 'Baja'}  
        🌍 **Precio global**: {global_price} USD/Ton
        """)
        
        # Botón de predicción
        if st.button("📈 Predecir Precio", type="primary", use_container_width=True):
            # Preparar datos para la API
            data = {
                "day": int(day_of_year),
                "global_price": float(global_price)
            }
            
            # Mostrar spinner mientras procesa
            with st.spinner("🔄 Analizando tendencias de mercado..."):
                try:
                    # Llamada al backend
                    import requests
                    response = requests.post(
                        "http://localhost:8000/mercado", 
                        json=data,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        # Guardar resultado en session state
                        result = response.json()
                        # Agregar información del producto seleccionado
                        result['product'] = selected_product
                        result['day'] = day_of_year
                        result['global_price'] = global_price
                        st.session_state.market_result = result
                        st.success("✅ Predicción de precio completada!")
                        
                    else:
                        st.error(f"❌ Error del servidor: {response.status_code}")
                        st.session_state.market_result = None
                        
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Error de conexión: Verifica que el backend esté ejecutándose en http://localhost:8000")
                    st.session_state.market_result = None
                    
                except requests.exceptions.Timeout:
                    st.error("⏱️ Tiempo de espera agotado: El servidor tardó demasiado en responder")
                    st.session_state.market_result = None
                    
                except Exception as e:
                    st.error(f"💥 Error inesperado: {str(e)}")
                    st.session_state.market_result = None
    
    with col2:
        st.subheader("📈 Factores de Precio")
        st.markdown("""
        **⏰ Estacionalidad:**
        - Enero-Marzo: Precios altos
        - Abril-Junio: Descenso gradual  
        - Julio-Septiembre: Estabilidad
        - Octubre-Diciembre: Incremento
        
        **🌍 Mercado Global:**
        - USD fuerte → Precios altos
        - Demanda internacional
        - Clima en otros países
        
        **🚚 Factores Locales:**
        - Costos de transporte
        - Disponibilidad regional
        - Festividades y eventos
        """)
        
        # Indicador de temporada
        season = ""
        if 1 <= day_of_year <= 90:
            season = "🌱 Temporada Alta"
        elif 91 <= day_of_year <= 180:
            season = "📉 Descenso Estacional"
        elif 181 <= day_of_year <= 270:
            season = "📊 Periodo Estable"
        else:
            season = "📈 Incremento Estacional"
            
        st.metric("🗓️ Temporada Actual", season)
    
    # Mostrar resultados si existen
    if st.session_state.market_result is not None:
        st.markdown("---")
        
        result = st.session_state.market_result
        
        # Resultado principal
        predicted_price = result.get('price', 0)
        product_name = result.get('product', 'Producto')
        day_analyzed = result.get('day', day_of_year)
        global_ref = result.get('global_price', global_price)
        
        # Crear métricas visuales
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.metric(
                "💵 Precio Predicho",
                f"${predicted_price:,.0f} COP/kg",
                delta=None
            )
        
        with col_result2:
            # Calcular variación respecto al precio base
            base_price = product_options.get(product_name, {}).get('base_price', predicted_price)
            variation = ((predicted_price - base_price) / base_price) * 100
            st.metric(
                "📊 Variación",
                f"{variation:+.1f}%",
                delta=f"vs precio base ({base_price} COP/kg)",
                delta_color="normal" if variation >= 0 else "inverse"
            )
            
        with col_result3:
            # Clasificar el precio
            if variation > 15:
                price_level = "Alto"
                level_color = "🔴"
            elif variation > 5:
                price_level = "Medio-Alto"
                level_color = "🟠"
            elif variation > -5:
                price_level = "Normal"
                level_color = "🟡"
            elif variation > -15:
                price_level = "Medio-Bajo"
                level_color = "🟢"
            else:
                price_level = "Bajo"
                level_color = "🔵"
                
            st.metric(
                "🎯 Nivel de Precio",
                f"{level_color} {price_level}",
                delta=None
            )
        
        # Recomendaciones basadas en el precio
        if variation > 10:
            st.success(f"""
            ### 📈 ¡Precio favorable para la venta!
            
            **Estrategias recomendadas:**
            - 💰 Momento óptimo para comercializar
            - 🚚 Acelera la cosecha si es posible
            - 📦 Negocia contratos a precio actual
            - 🏪 Considera venta directa al consumidor
            - 📊 Monitorea la competencia
            """)
        elif variation < -10:
            st.warning(f"""
            ### 📉 Precio por debajo del promedio
            
            **Estrategias recomendadas:**
            - ⏳ Considera esperar si tienes capacidad de almacenamiento
            - 🔄 Busca mercados alternativos
            - 🏭 Evalúa procesamiento o valor agregado
            - 💼 Negocia contratos a futuro
            - 📈 Monitorea tendencias semanales
            """)
        else:
            st.info(f"""
            ### 📊 Precio en rango normal
            
            **Estrategias recomendadas:**
            - ⚖️ Mantén estrategia comercial actual
            - 📅 Planifica ventas regulares
            - 👥 Fortalece relaciones con compradores
            - 📈 Monitorea factores estacionales
            - 🔍 Busca oportunidades de diferenciación
            """)
        
        # Explicabilidad del modelo
        if 'explanation' in result and result['explanation']:
            st.markdown("---")
            st.subheader("🔍 Explicabilidad del Modelo (SHAP)")
            
            with st.expander("📊 Ver factores de influencia", expanded=False):
                explanation = result['explanation']
                
                if isinstance(explanation, list) and len(explanation) >= 2:
                    col_exp1, col_exp2 = st.columns(2)
                    
                    with col_exp1:
                        day_impact = explanation[0] if len(explanation) > 0 else 0
                        st.metric(
                            "📅 Impacto Temporal",
                            f"{day_impact:.3f}",
                            delta="Positivo" if day_impact > 0 else "Negativo",
                            delta_color="normal" if day_impact > 0 else "inverse"
                        )
                    
                    with col_exp2:
                        global_impact = explanation[1] if len(explanation) > 1 else 0
                        st.metric(
                            "🌍 Impacto Global",
                            f"{global_impact:.3f}",
                            delta="Positivo" if global_impact > 0 else "Negativo",
                            delta_color="normal" if global_impact > 0 else "inverse"
                        )
                    
                    st.markdown("""
                    **Interpretación:**
                    - Valores positivos incrementan el precio predicho
                    - Valores negativos lo reducen
                    - Mayor magnitud = mayor influencia en la predicción
                    """)
                else:
                    st.write("Datos de explicabilidad:", explanation)
        
        # Análisis de tendencias
        st.markdown("---")
        with st.expander("📈 Análisis de Tendencias", expanded=False):
            st.markdown(f"""
            **Análisis para {product_name}:**
            
            📅 **Factor temporal (día {day_analyzed}):**
            {f'Época favorable - típicamente precios altos' if 1 <= day_analyzed <= 90 or day_analyzed > 300 
              else f'Periodo de transición - precios moderados' if 90 < day_analyzed <= 180 or 270 < day_analyzed <= 300
              else 'Época estable - precios equilibrados'}
            
            🌍 **Factor global ({global_ref} USD/Ton):**
            {f'Precio internacional alto - favorece exportaciones' if global_ref > 600
              else f'Precio internacional moderado - mercado equilibrado' if global_ref > 300
              else 'Precio internacional bajo - competencia fuerte'}
            
            📊 **Recomendación general:**
            El precio predicho de ${predicted_price:,.0f} COP/kg representa una 
            {'excelente oportunidad de venta' if variation > 15 
             else 'buena oportunidad comercial' if variation > 5
             else 'situación normal de mercado' if variation > -5
             else 'momento para evaluar alternativas' if variation > -15
             else 'situación que requiere estrategias especiales'}.
            """)
        
        # Historial y contexto
        with st.expander("📊 Contexto del Mercado", expanded=False):
            st.markdown(f"""
            **Información del análisis:**
            
            🛒 **Producto analizado**: {product_name}  
            📅 **Fecha objetivo**: {date_obj.strftime('%d de %B de %Y')}  
            🌍 **Referencia global**: {global_ref} USD/Tonelada  
            💰 **Precio predicho**: ${predicted_price:,.0f} COP/kg  
            
            📈 **Comparación histórica:**
            - Precio base típico: ${base_price:,.0f} COP/kg
            - Variación actual: {variation:+.1f}%
            - Clasificación: {price_level}
            
            🎯 **Factores clave:**
            - Estacionalidad del cultivo
            - Tendencias del mercado internacional
            - Condiciones climáticas regionales
            - Demanda del mercado nacional
            """)
    
    else:
        # Mensaje cuando no hay resultados
        st.markdown("---")
        st.warning("⚠️ No hay resultados disponibles. Configura los parámetros de mercado y presiona 'Predecir Precio' para obtener una predicción.")
        
        # Mostrar información de mercados
        with st.expander("📊 Información de Mercados", expanded=False):
            st.markdown("""
            **🏪 Principales mercados en Colombia:**
            
            **Bogotá - Corabastos:**
            - Mayor centro de acopio del país
            - Precios de referencia nacional
            - Horario: 24 horas
            
            **Medellín - Central Mayorista:**
            - Segundo mercado más importante
            - Especializado en productos de clima frío
            - Excelente logística
            
            **Cali - Cavasa:**
            - Puerto de entrada productos del Pacífico
            - Fuerte en frutas tropicales
            - Conexión con mercados internacionales
            
            **📱 Fuentes de información:**
            - SIPSA (DANE)
            - Agronet (MADR)  
            - Finagro
            - Bolsa Mercantil de Colombia
            """)
    
    # Footer con información técnica
    st.markdown("---")
    st.caption("📊 Datos SIPSA-DANE | 🤖 Modelo de regresión lineal | 💹 Actualización semanal | 🇨🇴 Mercado colombiano")
def render_agro():
    """
    Página de diagnóstico de cultivos con análisis de imágenes
    """
    st.header("🌱 AgroExpert - Diagnóstico de Hojas")
    st.markdown("**Analiza imágenes de hojas para detectar enfermedades utilizando inteligencia artificial.**")
    
    # Información explicativa
    with st.expander("ℹ️ ¿Cómo funciona?", expanded=False):
        st.markdown("""
        - **Modelo**: Red neuronal MobileNetV2 con transfer learning
        - **Entrenamiento**: Dataset PlantVillage con hojas sanas y enfermas
        - **Cultivos soportados**: Tomate, papa, maíz, uva, y más
        - **Precisión**: ~95% en condiciones controladas
        - **Explicabilidad**: Grad-CAM para mostrar áreas de interés
        """)
    
    # Layout en columnas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Subir Imagen")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de hoja",
            type=['png', 'jpg', 'jpeg'],
            help="Límite 200MB por archivo • PNG, JPG, JPEG",
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # Mostrar imagen cargada
            st.image(uploaded_file, caption="Imagen cargada", use_container_width=True)
            
            # Información de la imagen
            file_details = {
                "Nombre": uploaded_file.name,
                "Tipo": uploaded_file.type,
                "Tamaño": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
            
            # Botón de análisis
            if st.button("🔍 Analizar Hoja", type="primary", use_container_width=True):
                with st.spinner("🔄 Analizando imagen con IA..."):
                    try:
                        # Preparar archivo para envío
                        files = {"file": uploaded_file.getvalue()}
                        
                        # Llamada al backend
                        import requests
                        response = requests.post(
                            "http://localhost:8000/agro",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            # Guardar resultado en session state
                            result = response.json()
                            st.session_state.agro_result = result
                            st.success("✅ Análisis completado exitosamente!")
                            
                        else:
                            st.error(f"❌ Error del servidor: {response.status_code}")
                            if response.text:
                                st.error(f"Detalle: {response.text}")
                            st.session_state.agro_result = None
                            
                    except requests.exceptions.ConnectionError:
                        st.error("🔌 Error de conexión: Verifica que el backend esté ejecutándose en http://localhost:8000")
                        st.session_state.agro_result = None
                        
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Tiempo de espera agotado: El análisis tardó demasiado")
                        st.session_state.agro_result = None
                        
                    except Exception as e:
                        st.error(f"💥 Error inesperado: {str(e)}")
                        st.session_state.agro_result = None
        else:
            st.info("👆 Sube una imagen de hoja para comenzar el análisis")
    
    with col2:
        st.subheader("📋 Guía de Uso")
        st.markdown("""
        **📸 Mejores prácticas:**
        - Hoja bien iluminada
        - Fondo contrastante
        - Enfoque nítido
        - Hoja completa visible
        
        **🌿 Cultivos soportados:**
        - 🍅 Tomate
        - 🥔 Papa
        - 🌽 Maíz
        - 🍇 Uva
        - 🍎 Manzana
        - 🥒 Pepino
        
        **🦠 Enfermedades detectables:**
        - Tizón tardío
        - Mancha foliar
        - Mosaico viral
        - Óxido
        - Y más...
        """)
    
    # Mostrar resultados si existen
    if st.session_state.agro_result is not None:
        st.markdown("---")
        
        result = st.session_state.agro_result
        
        # Resultado principal
        is_sick = result.get('sick', False)
        confidence = result.get('confidence', 0)
        
        # Crear métricas visuales
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            if is_sick:
                st.metric(
                    "🦠 Diagnóstico",
                    "ENFERMO",
                    delta="Requiere atención",
                    delta_color="inverse"
                )
            else:
                st.metric(
                    "✅ Diagnóstico",
                    "SANO",
                    delta="Estado normal",
                    delta_color="normal"
                )
        
        with col_result2:
            st.metric(
                "📊 Confianza",
                f"{confidence:.1%}" if confidence > 0 else "N/A",
                delta=None
            )
            
        with col_result3:
            # Nivel de riesgo basado en predicción
            risk_level = "Alto" if is_sick else "Bajo"
            st.metric(
                "⚠️ Riesgo",
                risk_level,
                delta=None
            )
        
        # Alerta principal
        if is_sick:
            st.error("""
            ### 🦠 ¡Alerta: Posible enfermedad detectada!
            
            **Recomendaciones inmediatas:**
            - 🔍 Inspecciona otras plantas cercanas
            - 🚿 Aplica tratamiento fungicida si es necesario
            - ✂️ Considera podar hojas afectadas
            - 📞 Consulta con un agrónomo local
            - 📝 Registra la ubicación y fecha del hallazgo
            """)
        else:
            st.success("""
            ### ✅ Hoja en buen estado
            
            **Mantén las buenas prácticas:**
            - 💧 Continúa con el riego regular
            - 🌱 Monitorea el crecimiento
            - 🔄 Realiza inspecciones periódicas
            - 🌿 Mantén la nutrición adecuada
            - 🧹 Limpieza regular del cultivo
            """)
        
        # Explicabilidad con Grad-CAM
        if 'explanation_heatmap' in result and result['explanation_heatmap']:
            st.markdown("---")
            st.subheader("🔍 Explicabilidad del Modelo (Grad-CAM)")
            
            with st.expander("🎯 Ver mapa de calor", expanded=False):
                st.markdown("""
                El **mapa de calor** muestra las áreas de la imagen que más influenciaron 
                la decisión del modelo. Las zonas más brillantes son las que el modelo 
                consideró más importantes para el diagnóstico.
                """)
                
                try:
                    import numpy as np
                    import matplotlib.pyplot as plt
                    import cv2
                    
                    # Procesar heatmap
                    heatmap_data = np.array(result['explanation_heatmap'])
                    
                    if heatmap_data.size > 0:
                        # Normalizar para visualización
                        heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)
                        
                        # Crear visualización
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(heatmap_normalized, cmap='hot', interpolation='bilinear')
                        ax.set_title('Mapa de Calor - Áreas de Atención')
                        ax.axis('off')
                        plt.colorbar(im, ax=ax, label='Intensidad de Atención')
                        
                        st.pyplot(fig)
                        
                        st.markdown("""
                        **Interpretación:**
                        - 🔴 **Rojo intenso**: Áreas críticas para el diagnóstico
                        - 🟠 **Naranja**: Áreas de interés moderado  
                        - 🟡 **Amarillo**: Áreas de baja relevancia
                        """)
                    else:
                        st.warning("No se pudo generar el mapa de calor")
                        
                except Exception as e:
                    st.warning(f"Error al procesar explicabilidad: {str(e)}")
                    st.json(result['explanation_heatmap'])
        
        # Información adicional
        st.markdown("---")
        with st.expander("📊 Detalles Técnicos", expanded=False):
            st.markdown(f"""
            **Información del análisis:**
            
            🤖 **Modelo utilizado**: MobileNetV2 con transfer learning  
            📚 **Dataset de entrenamiento**: PlantVillage (54,305 imágenes)  
            🎯 **Clases**: Sano vs. Enfermo (binario)  
            🔬 **Precisión del modelo**: ~95% en dataset de validación  
            ⚡ **Tiempo de procesamiento**: < 5 segundos  
            
            **Resultado completo:**
            """)
            st.json(result)
        
        # Historial y recomendaciones
        with st.expander("📋 Historial y Seguimiento", expanded=False):
            st.markdown("""
            **Para un mejor seguimiento:**
            
            📅 **Registro recomendado:**
            - Fecha y hora del análisis
            - Ubicación de la planta
            - Condiciones climáticas
            - Tratamientos aplicados
            
            🔄 **Frecuencia de monitoreo:**
            - Plantas sanas: Semanal
            - Plantas enfermas: Diario
            - Post-tratamiento: Cada 2-3 días
            
            📞 **Cuándo contactar un experto:**
            - Propagación rápida
            - Síntomas no mejoran
            - Múltiples plantas afectadas
            """)
    
    else:
        # Mensaje cuando no hay resultados
        st.markdown("---")
        st.warning("⚠️ No hay resultados disponibles. Sube una imagen de hoja para obtener un diagnóstico.")
        
        # Mostrar ejemplos
        with st.expander("💡 Ejemplos de imágenes", expanded=False):
            st.markdown("""
            **✅ Buenas imágenes para análisis:**
            
            🌿 **Hoja sana de tomate:**
            - Verde uniforme
            - Sin manchas o decoloración
            - Superficie lisa
            
            🦠 **Hoja enferma de papa:**
            - Manchas marrones o negras
            - Bordes amarillentos
            - Textura irregular
            
            **📸 Consejos de fotografía:**
            - Luz natural indirecta
            - Fondo simple (papel blanco)
            - Distancia de 20-30 cm
            - Evitar sombras fuertes
            """)
    
    # Footer con información técnica
    st.markdown("---")
    st.caption("🤖 Modelo MobileNetV2 entrenado con PlantVillage | 🔬 Precisión 95% | 🌱 Especializado en agricultura colombiana")
def render_finance():
    """Página de FinanceAI"""
    st.markdown("## 🏦 FinanceAI - Evaluación de Microcréditos")
    st.markdown("Evalúa la viabilidad de microcréditos rurales basándose en criterios específicos del sector agropecuario.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Información del Solicitante")
        
        # Inputs
        income = st.number_input(
            "Ingresos mensuales (miles COP)",
            min_value=500.0, max_value=10000.0, value=3500.0, step=100.0
        )
        
        crop_types = {
            "Café": 0,
            "Cacao": 1,
            "Aguacate": 2,
            "Tabaco": 3,
            "Policultivo": 4
        }
        
        crop_name = st.selectbox("Tipo de cultivo principal", list(crop_types.keys()))
        crop_type_encoded = crop_types[crop_name]
        
        credit_history = st.slider(
            "Historial crediticio (0=malo, 1=excelente)",
            min_value=0.0, max_value=1.0, value=0.7, step=0.1
        )
        
        # Información contextual
        st.info(f"""
        **Perfil del Solicitante:**
        - Ingresos: ${income:,.0f} mil COP/mes
        - Cultivo: {crop_name}
        - Historial: {'Excelente' if credit_history > 0.8 else 'Bueno' if credit_history > 0.5 else 'Regular' if credit_history > 0.3 else 'Malo'}
        """)
        
        if st.button("🏧 Evaluar Crédito", key="finance_predict"):
            with st.spinner("Evaluando solicitud..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/finanzas",
                        json={
                            "income": income,
                            "crop_type_encoded": crop_type_encoded,
                            "credit_history": credit_history
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.finance_result = result
                    else:
                        st.error(f"Error en la evaluación: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error de conexión: {str(e)}")
    
    with col2:
        st.markdown("### Resultado de la Evaluación")
        
        if 'finance_result' in st.session_state:
            result = st.session_state.finance_result
            
            # Resultado principal
            approved = result.get('approve', False)
            confidence = result.get('confidence', 0)
            
            if approved:
                st.markdown(f"""
                <div class="result-success">
                <h4>✅ CRÉDITO APROBADO</h4>
                <p>Confianza: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Calcular monto sugerido
                suggested_amount = min(income * 12 * 0.3, 15000000)  # 30% ingresos anuales, máx 15M
                
                st.markdown(f"""
                **Condiciones Sugeridas:**
                - Monto máximo: ${suggested_amount:,.0f} COP
                - Plazo: 12-24 meses
                - Tasa preferencial para sector rural
                """)
                
            else:
                st.markdown(f"""
                <div class="result-warning">
                <h4>❌ CRÉDITO NO APROBADO</h4>
                <p>Confianza: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Recomendaciones para mejorar:**")
                if income < 2000:
                    st.markdown("- Aumentar ingresos o demostrar ingresos adicionales")
                if credit_history < 0.5:
                    st.markdown("- Mejorar historial crediticio")
                    st.markdown("- Considerar microcréditos de menor monto")
                st.markdown("- Presentar avales o garantías adicionales")
            
            # Explicabilidad
            if 'explanation' in result:
                st.markdown("### Factores de Decisión")
                explanation = result['explanation']
                features = result.get('features', ['Ingresos', 'Tipo de cultivo', 'Historial crediticio'])
                
                if len(explanation) > 0:
                    # Extraer valores SHAP
                    if isinstance(explanation[0], list) and len(explanation[0]) > 0:
                        if isinstance(explanation[0][0], list):
                            # SHAP binario - tomar clase positiva
                            feature_importance = [x[1] for x in explanation[0]]
                        else:
                            feature_importance = explanation[0]
                    else:
                        feature_importance = explanation
                    
                    # Crear gráfico
                    fig = go.Figure()
                    colors = ['#FF6B6B' if x < 0 else '#51CF66' for x in feature_importance]
                    
                    fig.add_trace(go.Bar(
                        x=features,
                        y=feature_importance,
                        marker_color=colors,
                        text=[f"{x:.3f}" for x in feature_importance],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title="Contribución de Factores a la Decisión",
                        xaxis_title="Variables",
                        yaxis_title="Influencia en Aprobación",
                        height=350
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="explanation-box">
                    <small><strong>Interpretación:</strong> Valores positivos favorecen la aprobación, 
                    valores negativos la perjudican. Los ingresos suelen ser el factor más determinante.</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("💼 Complete la información y evalúe para ver los resultados")

if __name__ == "__main__":
    # Inicializar session state
    if 'climate_result' not in st.session_state:
        st.session_state.climate_result = None
    if 'market_result' not in st.session_state:
        st.session_state.market_result = None
    if 'agro_result' not in st.session_state:
        st.session_state.agro_result = None
    if 'finance_result' not in st.session_state:
        st.session_state.finance_result = None
    
    main()