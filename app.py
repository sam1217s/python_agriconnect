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
    """Página de ClimateAI"""
    st.markdown("## 🌦️ ClimateAI - Predicción Climática")
    st.markdown("Predice la probabilidad de lluvia basándose en las condiciones de temperatura y humedad.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Datos de Entrada")
        
        # Inputs
        temperature = st.slider("Temperatura (°C)", 
                               min_value=10.0, max_value=40.0, value=25.0, step=0.5)
        
        humidity = st.slider("Humedad (%)", 
                            min_value=30.0, max_value=100.0, value=70.0, step=1.0)
        
        # Información contextual
        st.info(f"""
        **Condiciones Actuales:**
        - Temperatura: {temperature}°C
        - Humedad: {humidity}%
        """)
        
        if st.button("🔍 Predecir Clima", key="climate_predict"):
            with st.spinner("Analizando condiciones climáticas..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/clima",
                        json={"temp": temperature, "humidity": humidity},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Guardar en session state
                        st.session_state.climate_result = result
                        
                    else:
                        st.error(f"Error en la predicción: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error de conexión: {str(e)}")
    
    with col2:
        st.markdown("### Resultados")
        
        if 'climate_result' in st.session_state:
            result = st.session_state.climate_result
            
            # Resultado principal
            rain_prediction = result.get('rain', False)
            confidence = result.get('confidence', 0)
            
            if rain_prediction:
                st.markdown(f"""
                <div class="result-warning">
                <h4>☔ Predicción: LLUVIA</h4>
                <p>Confianza: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Recomendaciones:**")
                st.markdown("- Proteger cultivos sensibles al agua")
                st.markdown("- Revisar sistemas de drenaje")
                st.markdown("- Postponer aplicaciones foliares")
                
            else:
                st.markdown(f"""
                <div class="result-success">
                <h4>☀️ Predicción: SECO</h4>
                <p>Confianza: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Recomendaciones:**")
                st.markdown("- Considerar riego si es necesario")
                st.markdown("- Buen momento para aplicaciones foliares")
                st.markdown("- Condiciones favorables para cosecha")
            
            # Explicabilidad
            if 'explanation' in result:
                st.markdown("### Explicabilidad SHAP")
                explanation = result['explanation']
                features = result.get('features', ['Temperatura', 'Humedad'])
                
                if len(explanation) > 0 and len(explanation[0]) == 2:
                    feature_importance = explanation[0]
                    
                    # Crear gráfico de importancia
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=features,
                        y=feature_importance,
                        marker_color=['#FF6B6B' if x < 0 else '#51CF66' for x in feature_importance]
                    ))
                    
                    fig.update_layout(
                        title="Importancia de Factores",
                        xaxis_title="Variables",
                        yaxis_title="Contribución a la Predicción",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="explanation-box">
                    <small><strong>Interpretación:</strong> Valores positivos aumentan la probabilidad de lluvia, 
                    valores negativos la disminuyen.</small>
                    </div>
                    """, unsafe_allow_html=True)

def render_market():
    """Página de MarketAI"""
    st.markdown("## 💰 MarketAI - Predicción de Precios")
    st.markdown("Proyecta precios de productos agrícolas basándose en tendencias estacionales y precios globales.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Datos de Entrada")
        
        # Inputs
        day_of_year = st.slider("Día del año", 
                               min_value=1, max_value=365, value=250)
        
        global_price = st.slider("Precio global de referencia (USD)", 
                                min_value=100.0, max_value=800.0, value=350.0, step=10.0)
        
        # Mostrar fecha aproximada
        try:
            date = datetime(2025, 1, 1).replace(day=1) + pd.Timedelta(days=day_of_year-1)
            st.info(f"**Fecha aproximada:** {date.strftime('%d/%m/%Y')}")
        except:
            st.info(f"**Día del año:** {day_of_year}")
        
        if st.button("📊 Predecir Precio", key="market_predict"):
            with st.spinner("Analizando mercado..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/mercado",
                        json={"day": day_of_year, "global_price": global_price},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.market_result = result
                    else:
                        st.error(f"Error en la predicción: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error de conexión: {str(e)}")
    
    with col2:
        st.markdown("### Resultados")
        
        if 'market_result' in st.session_state:
            result = st.session_state.market_result
            
            # Resultado principal
            predicted_price = result.get('price', 0)
            currency = result.get('currency', 'COP/kg')
            
            st.markdown(f"""
            <div class="result-success">
            <h4>💵 Precio Proyectado</h4>
            <h2>${predicted_price:,.0f} {currency}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Contexto de precios
            st.markdown("### Contexto del Mercado")
            
            # Comparación con rangos típicos
            price_ranges = {
                "Café": (8000, 15000),
                "Cacao": (3000, 6000),
                "Aguacate": (2500, 4500),
                "Yuca": (1500, 3500)
            }
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Rangos de Precios Típicos:**")
                for product, (min_price, max_price) in price_ranges.items():
                    in_range = min_price <= predicted_price <= max_price
                    status = "✅" if in_range else "📊"
                    st.markdown(f"{status} {product}: ${min_price:,} - ${max_price:,}")
            
            with col_b:
                # Recomendaciones
                st.markdown("**Recomendaciones:**")
                if predicted_price > 4000:
                    st.markdown("- Precio alto: Considerar venta")
                    st.markdown("- Evaluar calidad del producto")
                elif predicted_price > 2500:
                    st.markdown("- Precio moderado: Analizar tendencias")
                    st.markdown("- Esperar mejores condiciones si es posible")
                else:
                    st.markdown("- Precio bajo: Evaluar costos")
                    st.markdown("- Considerar almacenamiento temporal")
            
            # Explicabilidad
            if 'explanation' in result:
                st.markdown("### Análisis de Factores")
                explanation = result['explanation']
                features = result.get('features', ['Día del año', 'Precio global'])
                
                if len(explanation) > 0:
                    feature_importance = explanation[0] if isinstance(explanation[0], list) else explanation
                    
                    # Crear gráfico
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=features,
                        y=feature_importance,
                        marker_color=['#FF9999', '#99CCFF']
                    ))
                    
                    fig.update_layout(
                        title="Contribución de Factores al Precio",
                        xaxis_title="Variables",
                        yaxis_title="Influencia en el Precio",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def render_agro():
    """Página de AgroExpert"""
    st.markdown("## 🌿 AgroExpert - Diagnóstico de Hojas")
    st.markdown("Analiza imágenes de hojas para detectar enfermedades utilizando inteligencia artificial.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Subir Imagen")
        
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de hoja",
            type=['png', 'jpg', 'jpeg'],
            help="Formatos soportados: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Mostrar imagen
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_column_width=True)
            
            # Información de la imagen
            st.markdown(f"""
            **Información del archivo:**
            - Nombre: {uploaded_file.name}
            - Tamaño: {len(uploaded_file.getvalue())} bytes
            - Dimensiones: {image.size}
            """)
            
            if st.button("🔬 Analizar Hoja", key="agro_predict"):
                with st.spinner("Analizando imagen..."):
                    try:
                        # Preparar archivo para envío
                        files = {"file": uploaded_file.getvalue()}
                        
                        response = requests.post(
                            f"{API_BASE_URL}/agro",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.agro_result = result
                            st.session_state.agro_image = image
                        else:
                            st.error(f"Error en el análisis: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Error de conexión: {str(e)}")
        else:
            st.info("👆 Sube una imagen de hoja para comenzar el análisis")
    
    with col2:
        st.markdown("### Resultados del Diagnóstico")
        
        if 'agro_result' in st.session_state:
            result = st.session_state.agro_result
            
            # Resultado principal
            is_sick = result.get('sick', False)
            confidence = result.get('confidence', 0)
            
            if is_sick:
                st.markdown(f"""
                <div class="result-warning">
                <h4>🦠 Diagnóstico: ENFERMEDAD DETECTADA</h4>
                <p>Confianza: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Recomendaciones:**")
                st.markdown("- Aislar plantas afectadas")
                st.markdown("- Consultar con un agrónomo")
                st.markdown("- Aplicar tratamiento específico")
                st.markdown("- Monitorear cultivos cercanos")
                
            else:
                st.markdown(f"""
                <div class="result-success">
                <h4>✅ Diagnóstico: HOJA SALUDABLE</h4>
                <p>Confianza: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Recomendaciones:**")
                st.markdown("- Mantener prácticas actuales")
                st.markdown("- Continuar monitoreo regular")
                st.markdown("- Seguir calendario de nutrición")
            
            # Explicabilidad con Grad-CAM
            if 'explanation_heatmap' in result:
                st.markdown("### Mapa de Atención (Grad-CAM)")
                
                heatmap_data = np.array(result['explanation_heatmap'])
                
                if heatmap_data.size > 0:
                    # Crear visualización del heatmap
                    fig = px.imshow(
                        heatmap_data,
                        color_continuous_scale='Viridis',
                        title="Regiones de Importancia para el Diagnóstico"
                    )
                    
                    fig.update_layout(
                        height=400,
                        xaxis_title="Píxeles (X)",
                        yaxis_title="Píxeles (Y)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="explanation-box">
                    <small><strong>Interpretación:</strong> Las áreas más brillantes son las regiones 
                    de la imagen que más influyeron en el diagnóstico del modelo.</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("📸 Sube y analiza una imagen para ver los resultados aquí")

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