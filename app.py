import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Objetos",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS minimalistas
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #262730;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .section-header {
        font-size: 1.3rem;
        color: #262730;
        margin: 2rem 0 1rem 0;
        font-weight: 400;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 300;
        color: #262730;
    }
    .stCamera > div > div {
        border-radius: 8px;
    }
    .stImage > img {
        border-radius: 8px;
    }
    div[data-testid="stSidebar"] {
        background-color: #fafafa;
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

# Encabezado
st.markdown('<h1 class="main-header">🔍 Detección de Objetos</h1>', unsafe_allow_html=True)

# Descripción minimalista
st.markdown("Detección de objetos en tiempo real usando YOLOv5")
st.markdown("---")

# Cargar modelo
with st.spinner("Cargando modelo..."):
    model = load_yolov5_model()

if model:
    # Sidebar - Configuración
    st.sidebar.markdown('<div class="section-header">Configuración</div>', unsafe_allow_html=True)
    
    # Parámetros principales
    model.conf = st.sidebar.slider('Umbral de confianza', 0.0, 1.0, 0.25, 0.01)
    model.iou = st.sidebar.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
    
    # Opciones avanzadas
    with st.sidebar.expander("Opciones avanzadas"):
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas', False)
            model.max_det = st.number_input('Máximo de detecciones', 10, 2000, 1000, 10)
        except:
            st.warning("Opciones limitadas")

    # Captura de imagen
    st.markdown('<div class="section-header">Capturar imagen</div>', unsafe_allow_html=True)
    picture = st.camera_input(" ", key="camera")
    
    if picture:
        # Procesar imagen
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Detección
        with st.spinner("Procesando..."):
            try:
                results = model(cv2_img)
            except Exception as e:
                st.error(f"Error en detección: {str(e)}")
                st.stop()
        
        # Mostrar resultados
        try:
            predictions = results.pred[0]
            categories = predictions[:, 5]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="section-header">Resultado</div>', unsafe_allow_html=True)
                results.render()
                st.image(cv2_img, channels='BGR', use_container_width=True)
            
            with col2:
                st.markdown('<div class="section-header">Estadísticas</div>', unsafe_allow_html=True)
                
                # Métricas
                total_objects = len(categories)
                label_names = model.names
                
                category_count = {}
                for category in categories:
                    category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                    category_count[category_idx] = category_count.get(category_idx, 0) + 1
                
                unique_categories = len(category_count)
                
                # Mostrar métricas
                col_metric1, col_metric2 = st.columns(2)
                with col_metric1:
                    st.markdown(f'<div class="metric-value">{total_objects}</div>', unsafe_allow_html=True)
                    st.caption("Objetos totales")
                
                with col_metric2:
                    st.markdown(f'<div class="metric-value">{unique_categories}</div>', unsafe_allow_html=True)
                    st.caption("Categorías")
                
                # Tabla de resultados
                if category_count:
                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        data.append({"Objeto": label, "Cantidad": count})
                    
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No se detectaron objetos")
                    
        except Exception as e:
            st.error(f"Error procesando resultados: {str(e)}")

else:
    st.error("No se pudo cargar el modelo")

# Pie de página
st.markdown("---")
st.caption("Detección de objetos con YOLOv5 • Streamlit")
