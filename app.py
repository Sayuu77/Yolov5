import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página
st.set_page_config(
    page_title="Vision AI - Detección de Objetos",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS mejorados
st.markdown("""
<style>
    /* Colores principales con mejor contraste */
    :root {
        --primary: #2563EB;
        --primary-dark: #2563EB;
        --secondary: #F8FAFC;
        --text: #0F172A;
        --text-light: #475569;
        --border: #E2E8F0;
        --success: #059669;
        --background: #FFFFFF;
    }
    
    .main-header {
        font-size: 2.8rem;
        color: var(--text);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-header {
        font-size: 1.4rem;
        color: var(--text);
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        padding-bottom: 0.5rem;
    }
    
    .section-header-white {
        font-size: 1.4rem;
        color: white !important;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: var(--background);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        text-align: center;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-light);
        font-weight: 500;
    }
    
    .camera-container {
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid var(--border);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .result-container {
        background: var(--background);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid var(--border);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .instructions-box {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .instructions-box h3 {
        color: white;
        margin-bottom: 1rem;
    }
    
    .instructions-box li {
        color: white;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    div[data-testid="stSidebar"] {
        background: var(--secondary);
        border-right: 1px solid var(--border);
    }
    
    .sidebar-header {
        font-size: 1.4rem;
        color: white !important;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        padding-bottom: 0.5rem;
    }
    
    /* Streamlit components customization */
    .stSlider > div > div > div {
        background: var(--primary);
    }
    
    .stProgress > div > div > div > div {
        background: var(--primary);
    }
    
    .stCamera > div > div {
        border-radius: 12px;
    }
    
    .stImage > img {
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    
    /* Text contrast improvements */
    .stMarkdown {
        color: var(--text);
    }
    
    .stCaption {
        color: var(--text-light) !important;
    }
    
    .stInfo {
        background-color: #EFF6FF;
        border-color: var(--primary);
    }
    
    /* Remove separator lines */
    hr {
        display: none !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    
    .dataframe th {
        background-color: var(--primary) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

# Header elegante
st.markdown('<h1 class="main-header">🎯 Vision AI</h1>', unsafe_allow_html=True)
st.markdown("### Detección de objetos en tiempo real con YOLOv5")

# Cargar modelo
with st.spinner("🔄 Inicializando modelo de IA..."):
    model = load_yolov5_model()

# Instrucciones en un lugar prominente
st.markdown("""
<div class="instructions-box">
    <h3>🚀 Cómo usar esta aplicación</h3>
    <ol>
        <li><strong>Configura los parámetros</strong> en el panel lateral izquierdo</li>
        <li><strong>Toma una foto</strong> usando la cámara debajo</li>
        <li><strong>Espera el análisis automático</strong> de la imagen</li>
        <li><strong>Revisa los resultados</strong> y estadísticas detalladas</li>
    </ol>
    <p><strong>💡 Tip:</strong> Ajusta el umbral de confianza para detectar más o menos objetos</p>
</div>
""", unsafe_allow_html=True)

if model:
    # Sidebar - Panel de control
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🎛️ Panel de Control</div>', unsafe_allow_html=True)
        
        # Parámetros principales
        st.markdown("**Parámetros de detección**")
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01,
                             help="Confianza requerida para considerar una detección válida")
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01,
                            help="Intersección sobre Unión para supresión de no máximos")
        
        # Mostrar valores actuales
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confianza", f"{model.conf:.2f}")
        with col2:
            st.metric("IoU", f"{model.iou:.2f}")
        
        # Opciones avanzadas
        with st.expander("⚙️ Configuración avanzada"):
            try:
                model.agnostic = st.checkbox('NMS class-agnostic', False)
                model.multi_label = st.checkbox('Múltiples etiquetas', False)
                model.max_det = st.number_input('Máx. detecciones', 10, 2000, 1000)
            except:
                st.info("Configuración básica activada")

    # Área principal - Cámara
    st.markdown('<div class="section-header-white">📷 Captura de Imagen</div>', unsafe_allow_html=True)
    
    # Cámara con contenedor estilizado
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)
    picture = st.camera_input("Toma una foto para analizar", key="camera")
    st.markdown('</div>', unsafe_allow_html=True)

    # Procesamiento de imagen
    if picture:
        # Convertir imagen
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Detección
        with st.spinner("🔍 Analizando imagen..."):
            try:
                results = model(cv2_img)
            except Exception as e:
                st.error(f"Error en el análisis: {str(e)}")
                st.stop()
        
        # Mostrar resultados
        try:
            predictions = results.pred[0]
            categories = predictions[:, 5]
            
            # Layout de resultados
            st.markdown('<div class="section-header">📊 Resultados del Análisis</div>', unsafe_allow_html=True)
            
            col_res1, col_res2 = st.columns([2, 1])
            
            with col_res1:
                st.markdown("**Imagen procesada**")
                results.render()
                st.image(cv2_img, channels='BGR', use_container_width=True)
            
            with col_res2:
                st.markdown("**Estadísticas de detección**")
                
                # Métricas principales
                total_objects = len(categories)
                label_names = model.names
                
                category_count = {}
                for category in categories:
                    category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                    category_count[category_idx] = category_count.get(category_idx, 0) + 1
                
                unique_categories = len(category_count)
                
                # Tarjetas de métricas
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Objetos detectados</div>
                    </div>
                    """.format(total_objects), unsafe_allow_html=True)
                
                with col_met2:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Categorías únicas</div>
                    </div>
                    """.format(unique_categories), unsafe_allow_html=True)
                
                # Tabla de detecciones
                st.markdown("**Detalles por categoría**")
                if category_count:
                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        data.append({"Objeto": label, "Cantidad": count})
                    
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Gráfico simple
                    st.markdown("**Distribución**")
                    chart_data = pd.DataFrame({
                        'Categoría': [label_names[cat] for cat in category_count.keys()],
                        'Cantidad': list(category_count.values())
                    })
                    st.bar_chart(chart_data.set_index('Categoría'))
                else:
                    st.info("No se detectaron objetos. Intenta ajustar los parámetros en el panel lateral.")
                    
        except Exception as e:
            st.error(f"Error procesando resultados: {str(e)}")

else:
    st.error("❌ No se pudo inicializar el modelo de IA")

# Footer elegante
st.markdown(
    """
    <div style='text-align: center; color: var(--text-light); padding: 2rem 0;'>
        <strong style='color: var(--text);'>Vision AI</strong> • Detección de objetos en tiempo real • Desarrollado con Streamlit y YOLOv5
    </div>
    """, 
    unsafe_allow_html=True
)
