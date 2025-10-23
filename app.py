import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .camera-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar el modelo YOLOv5 de manera compatible con versiones anteriores de PyTorch
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        # Importar yolov5
        import yolov5
        
        # Para versiones de PyTorch anteriores a 2.0, cargar directamente con weights_only=False
        # o usar el parámetro map_location para asegurar compatibilidad
        try:
            # Primer método: cargar con weights_only=False si la versión lo soporta
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            # Segundo método: si el primer método falla, intentar un enfoque más básico
            try:
                model = yolov5.load(model_path)
                return model
            except Exception as e:
                # Si todo falla, intentar cargar el modelo con torch directamente
                st.warning(f"Intentando método alternativo de carga...")
                
                # Modificar sys.path temporalmente para poder importar torch correctamente
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                
                # Cargar el modelo con torch directamente
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Instalar una versión compatible de PyTorch y YOLOv5:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. Asegúrate de tener el archivo del modelo en la ubicación correcta
        3. Si el problema persiste, intenta descargar el modelo directamente de torch hub
        """)
        return None

# Encabezado principal
st.markdown('<h1 class="main-header">🔍 Detección de Objetos en Tiempo Real</h1>', unsafe_allow_html=True)

# Información de la aplicación
st.markdown("""
<div class="info-box">
    <h3>📋 Acerca de esta aplicación</h3>
    <p>Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara. 
    Ajusta los parámetros en la barra lateral para personalizar la detección.</p>
</div>
""", unsafe_allow_html=True)

# Cargar el modelo
with st.spinner("🔄 Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

# Si el modelo se cargó correctamente, configuramos los parámetros
if model:
    # Sidebar para los parámetros de configuración
    st.sidebar.markdown('<h2 class="sub-header">⚙️ Parámetros de Configuración</h2>', unsafe_allow_html=True)
    
    # Ajustar parámetros del modelo
    with st.sidebar:
        st.markdown('<h3 class="sub-header">🔍 Configuración de detección</h3>', unsafe_allow_html=True)
        
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01, 
                              help="Establece el umbral mínimo de confianza para considerar una detección válida")
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01,
                             help="Establece el umbral de Intersección sobre Unión para la supresión de no máximos")
        
        # Mostrar valores actuales de manera visual
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confianza", f"{model.conf:.2f}")
        with col2:
            st.metric("IoU", f"{model.iou:.2f}")
        
        # Opciones adicionales
        st.markdown('<h3 class="sub-header">⚡ Opciones avanzadas</h3>', unsafe_allow_html=True)
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False,
                                        help="Habilita la supresión de no máximos agnóstica a clases")
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False,
                                           help="Permite múltiples etiquetas por caja delimitadora")
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10,
                                           help="Número máximo de detecciones a mostrar")
        except:
            st.warning("⚠️ Algunas opciones avanzadas no están disponibles con esta configuración")
    
    # Contenedor principal para la cámara y resultados
    main_container = st.container()
    
    with main_container:
        # Sección de captura de imagen
        st.markdown('<h2 class="sub-header">📷 Captura de Imagen</h2>', unsafe_allow_html=True)
        
        # Instrucciones para el usuario
        st.info("💡 **Instrucciones**: Haz clic en el botón de abajo para tomar una foto con tu cámara. La aplicación detectará automáticamente los objetos en la imagen.")
        
        # Capturar foto con la cámara
        with st.container():
            st.markdown('<div class="camera-container">', unsafe_allow_html=True)
            picture = st.camera_input("Capturar imagen", key="camera")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if picture:
            # Procesar la imagen capturada
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Realizar la detección
            with st.spinner("🔍 Detectando objetos en la imagen..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"❌ Error durante la detección: {str(e)}")
                    st.stop()
            
            # Parsear resultados
            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]
                
                # Mostrar resultados
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown('<h2 class="sub-header">🖼️ Imagen con Detecciones</h2>', unsafe_allow_html=True)
                    # Renderizar las detecciones
                    results.render()
                    # Mostrar imagen con las detecciones
                    st.image(cv2_img, channels='BGR', use_container_width=True, 
                            caption="Imagen procesada con detecciones de objetos")
                
                with col2:
                    st.markdown('<h2 class="sub-header">📊 Resultados de Detección</h2>', unsafe_allow_html=True)
                    
                    # Obtener nombres de etiquetas
                    label_names = model.names
                    
                    # Contar categorías
                    category_count = {}
                    for category in categories:
                        category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        if category_idx in category_count:
                            category_count[category_idx] += 1
                        else:
                            category_count[category_idx] = 1
                    
                    # Mostrar métricas generales
                    total_objects = len(categories)
                    unique_categories = len(category_count)
                    
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Total de Objetos</h3>
                            <h2 style="color: #1f77b4;">{total_objects}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Categorías Detectadas</h3>
                            <h2 style="color: #1f77b4;">{unique_categories}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Crear dataframe para mostrar resultados
                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                        data.append({
                            "Categoría": label,
                            "Cantidad": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })
                    
                    if data:
                        df = pd.DataFrame(data)
                        st.markdown("#### 📋 Detalles por Categoría")
                        st.dataframe(df, use_container_width=True)
                        
                        # Mostrar gráfico de barras
                        st.markdown("#### 📊 Distribución de Objetos Detectados")
                        st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    else:
                        st.warning("⚠️ No se detectaron objetos con los parámetros actuales.")
                        st.info("💡 Prueba a reducir el umbral de confianza en la barra lateral para aumentar la sensibilidad.")
            except Exception as e:
                st.error(f"❌ Error al procesar los resultados: {str(e)}")
                st.stop()
else:
    st.error("❌ No se pudo cargar el modelo. Por favor verifica las dependencias e inténtalo nuevamente.")
    st.stop()

# Información adicional y pie de página
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <h4>📚 Acerca de la aplicación</h4>
    <p>Esta aplicación utiliza YOLOv5 para detección de objetos en tiempo real. Desarrollada con Streamlit y PyTorch.</p>
    <p><small>YOLOv5 es un modelo de detección de objetos de última generación que ofrece un equilibrio entre velocidad y precisión.</small></p>
</div>
""", unsafe_allow_html=True)
