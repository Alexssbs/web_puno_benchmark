"""
Página de Metodología - Documentación del Pipeline
Benchmarking GFMs en Puno
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ============================================================
def show():
    """Función principal de la página de metodología"""
    
    st.title("📝 Metodología")
    st.markdown("""
    Documentación detallada del pipeline metodológico utilizado en el 
    benchmarking de Foundation Models para predicción de pobreza en Puno.
    """)
    
    # ============================================================
    # SECCIÓN 1: DIAGRAMA DEL PIPELINE
    # ============================================================
    st.markdown("---")
    st.markdown("### 🔄 Pipeline Metodológico")
    
    # Crear diagrama con plotly
    fig = go.Figure()
    
    # Definir nodos del pipeline
    nodos = [
        {"id": "GEE", "label": "Google Earth Engine", "x": 0, "y": 0, "color": "#2E86C1"},
        {"id": "Descarga", "label": "Descarga Imágenes", "x": 2, "y": 0, "color": "#3498DB"},
        {"id": "Preprocesamiento", "label": "Pre-procesamiento", "x": 4, "y": 0, "color": "#5DADE2"},
        {"id": "Embeddings", "label": "Extracción Embeddings", "x": 6, "y": 0, "color": "#85C1E9"},
        {"id": "Reduccion", "label": "Reducción Dimensional", "x": 8, "y": 0, "color": "#85C1E9"},
        {"id": "Regresion", "label": "Regresión", "x": 10, "y": 0, "color": "#F1948A"},
        {"id": "Evaluacion", "label": "Evaluación", "x": 12, "y": 0, "color": "#E74C3C"}
    ]
    
    # Agregar nodos al diagrama
    for nodo in nodos:
        fig.add_trace(go.Scatter(
            x=[nodo["x"]],
            y=[nodo["y"]],
            mode='markers+text',
            marker=dict(size=50, color=nodo["color"], symbol='square'),
            text=nodo["label"],
            textposition='bottom center',
            textfont=dict(size=12, color='black'),
            hoverinfo='text',
            hovertext=nodo["label"],
            name=''
        ))
    
    # Agregar flechas con anotaciones
    for i in range(len(nodos)-1):
        fig.add_annotation(
            x=(nodos[i]["x"] + nodos[i+1]["x"])/2,
            y=nodos[i]["y"] - 0.5,
            text="→",
            showarrow=False,
            font=dict(size=30, color="#666")
        )
    
    # Configurar layout
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-1, 14]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-2, 2]
        ),
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================
    # SECCIÓN 2: GENERACIÓN DEL DATASET
    # ============================================================
    st.markdown("---")
    st.markdown("### 🛰️ Generación del Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📋 Fuente de Datos
        
        - **Etiquetas**: Mapa de Pobreza Monetaria 2018 (INEI)
        - **Metodología**: Small Area Estimation
        - **Unidad**: Distritos (110 distritos en Puno)
        - **Variable objetivo**: Porcentaje de pobreza monetaria
        - **Rango**: 12.80% - 67.85%
        """)
    
    with col2:
        st.markdown("""
        #### 🛰️ Imágenes Satelitales
        
        - **Fuente**: Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED)
        - **Año**: 2018 (alineado con etiquetas)
        - **Filtro**: Nubes < 20%
        - **Composite**: Mediano anual
        - **Buffer**: 500 m alrededor del centro
        - **Bandas**: 12 bandas (B1-B12)
        - **Resolución**: 10 m
        """)
    
    # ============================================================
    # SECCIÓN 3: MODELOS EVALUADOS
    # ============================================================
    st.markdown("---")
    st.markdown("### 🧠 Modelos Evaluados")
    
    modelos = pd.DataFrame([
        {
            "Modelo": "ResNet50",
            "Tipo": "CNN Supervisada",
            "Arquitectura": "Residual Networks 50 capas",
            "Pre-entrenamiento": "ImageNet (1.2M imágenes)",
            "Dimensiones": "2048",
            "Bandas": "RGB (3)",
            "Ventaja": "Baseline sólido"
        },
        {
            "Modelo": "DINOv2",
            "Tipo": "ViT Self-Supervised",
            "Arquitectura": "Vision Transformer Base",
            "Pre-entrenamiento": "LVD-142M (1.4B imágenes)",
            "Dimensiones": "768",
            "Bandas": "RGB (3)",
            "Ventaja": "Self-supervised a gran escala"
        },
        {
            "Modelo": "ViT",
            "Tipo": "Vision Transformer",
            "Arquitectura": "ViT-Large (patch 16x16)",
            "Pre-entrenamiento": "ImageNet-21K",
            "Dimensiones": "1024",
            "Bandas": "RGB (3)",
            "Ventaja": "Arquitectura Transformer"
        },
        {
            "Modelo": "SatMAE",
            "Tipo": "Masked Autoencoder",
            "Arquitectura": "ViT-Base con codificación espectral",
            "Pre-entrenamiento": "fMoW-Temporal (1M+ imágenes)",
            "Dimensiones": "768",
            "Bandas": "10 (B2-B8A, B11-B12)",
            "Ventaja": "Diseñado para satélites"
        }
    ])
    
    st.dataframe(
        modelos.style
        .background_gradient(subset=['Dimensiones'], cmap='Blues'),
        use_container_width=True,
        hide_index=True
    )
    
    # Tarjetas de cada modelo
    st.markdown("#### 🏷️ Detalles de cada modelo")
    
    # ResNet50
    with st.expander("🔵 ResNet50 - Baseline Supervisado"):
        st.markdown("""
        **Arquitectura:** CNN con 50 capas (Residual Networks)
        
        **Pre-entrenamiento:** ImageNet (1.2M imágenes, 1000 clases)
        
        **Adaptación para satélite:**
        - Primera capa convolucional modificada: 3 → 12 canales
        - Nuevos canales inicializados promediando pesos RGB
        
        **Embedding:** 2048 dimensiones (última capa antes de clasificación)
        
        **Ventaja:** Baseline robusto y ampliamente utilizado
        """)
    
    # DINOv2
    with st.expander("🟣 DINOv2 - Self-Supervised Vision Transformer"):
        st.markdown("""
        **Arquitectura:** Vision Transformer Base
        
        **Pre-entrenamiento:** LVD-142M (1.4B imágenes, self-supervised)
        
        **Versión:** facebook/dinov2-base (HuggingFace)
        
        **Procesamiento:** 
        - AutoImageProcessor para normalización
        - Embedding del token CLS ([CLS] token)
        
        **Embedding:** 768 dimensiones
        
        **Ventaja:** Aprendizaje sin supervisión a gran escala
        """)
    
    # ViT
    with st.expander("🟢 ViT - Vision Transformer"):
        st.markdown("""
        **Arquitectura:** ViT-Large (patch 16x16)
        
        **Pre-entrenamiento:** ImageNet-21K
        
        **Versión:** google/vit-large-patch16-224
        
        **Normalización:** 
        - Percentiles (recorte 2% inferior y superior)
        - Adaptación a distribución ImageNet
        
        **Embedding:** 1024 dimensiones
        
        **Ventaja:** Arquitectura Transformer probada en imágenes
        """)
    
    # SatMAE
    with st.expander("🟠 SatMAE - Foundation Model Geoespacial"):
        st.markdown("""
        **Arquitectura:** Masked Autoencoder con codificación espectral
        
        **Checkpoint:** pretrain-vit-base-e199.pth
        
        **Pre-entrenamiento:** fMoW-Temporal (1M+ imágenes satelitales)
        
        **Características clave:**
        - Codificación posicional para dimensión espectral
        - Enmascaramiento independiente de parches
        - 10 bandas espectrales (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)
        
        **Embedding:** Token CLS (768 dimensiones)
        
        **Ventaja:** Diseñado específicamente para datos satelitales
        """)
    
    # ============================================================
    # SECCIÓN 4: PRE-PROCESAMIENTO
    # ============================================================
    st.markdown("---")
    st.markdown("### 🔧 Pre-procesamiento de Imágenes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📥 Post-descarga
        
        1. **Reemplazo de NaN e Inf**
           - `np.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)`
           - Preserva georreferenciación
        
        2. **Conversión de tipo**
           - float64 → float32
           - Optimiza almacenamiento
        
        3. **Normalización**
           - Reflectancia en [0, 1]
           - Valores ya normalizados de GEE
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Estadísticas del Dataset
        
        - **Total imágenes:** 110
        - **Imágenes con NaN:** 110 (100%)
        - **Total NaN:** 21.96% de píxeles
        - **Rango valores:** [0, 0.5932]
        - **Media:** 0.1388
        
        #### ⚙️ Extracción de Embeddings
        
        - **Tamaño de batch:** 
          - ResNet50: 16
          - DINOv2: 12
          - ViT: 4
          - SatMAE: 110 (directo)
        - **Tiempo SatMAE:** 19 segundos (5.61 img/seg)
        """)
    
    # ============================================================
    # SECCIÓN 5: REDUCCIÓN DIMENSIONAL
    # ============================================================
    st.markdown("---")
    st.markdown("### 📉 Técnicas de Reducción Dimensional")
    
    reduccion_data = pd.DataFrame([
        {
            "Técnica": "PCA",
            "Descripción": "Análisis de Componentes Principales",
            "Parámetros": "10-150 componentes",
            "Mejor Modelo": "SatMAE (50)",
            "Mejor R²": "0.0033"
        },
        {
            "Técnica": "UMAP",
            "Descripción": "Uniform Manifold Approximation",
            "Parámetros": "n_neighbors=15,30; min_dist=0.1,0.5",
            "Mejor Modelo": "DINOv2 (15,0.5)",
            "Mejor R²": "-0.0136"
        },
        {
            "Técnica": "KernelPCA",
            "Descripción": "PCA con kernels no lineales",
            "Parámetros": "kernel=rbf, poly",
            "Mejor Modelo": "DINOv2 (poly)",
            "Mejor R²": "-0.0155"
        },
        {
            "Técnica": "PLSRegression",
            "Descripción": "Mínimos Cuadrados Parciales",
            "Parámetros": "10-50 componentes",
            "Mejor Modelo": "DINOv2 (10)",
            "Mejor R²": "-0.9958"
        }
    ])
    
    st.dataframe(
        reduccion_data.style
        .background_gradient(subset=['Mejor R²'], cmap='RdYlGn_r'),
        use_container_width=True,
        hide_index=True
    )
    
    # ============================================================
    # SECCIÓN 6: REGRESORES
    # ============================================================
    st.markdown("---")
    st.markdown("### 📈 Regresores Evaluados")
    
    regresores_data = pd.DataFrame([
        {
            "Regresor": "Ridge",
            "Tipo": "Lineal Regularizado",
            "Mejor Modelo": "ViT",
            "R²": "-3.1708",
            "MAE": "13.157"
        },
        {
            "Regresor": "RandomForest",
            "Tipo": "Ensemble de Árboles",
            "Mejor Modelo": "ViT",
            "R²": "-0.1386",
            "MAE": "7.288"
        },
        {
            "Regresor": "XGBoost",
            "Tipo": "Gradient Boosting",
            "Mejor Modelo": "ResNet50",
            "R²": "-0.0444",
            "MAE": "—"
        },
        {
            "Regresor": "SVR (RBF)",
            "Tipo": "Support Vector Regression",
            "Mejor Modelo": "SatMAE",
            "R²": "0.0033",
            "MAE": "6.967"
        },
        {
            "Regresor": "ExtraTrees",
            "Tipo": "Extra Trees",
            "Mejor Modelo": "ViT",
            "R²": "-0.0917",
            "MAE": "—"
        },
        {
            "Regresor": "GradientBoosting",
            "Tipo": "Gradient Boosting",
            "Mejor Modelo": "ViT",
            "R²": "-0.0293",
            "MAE": "—"
        },
        {
            "Regresor": "CatBoost",
            "Tipo": "Gradient Boosting",
            "Mejor Modelo": "DINOv2",
            "R²": "-0.1380",
            "MAE": "—"
        }
    ])
    
    st.dataframe(
        regresores_data.style
        .background_gradient(subset=['R²'], cmap='RdYlGn_r'),
        use_container_width=True,
        hide_index=True
    )
    
    # ============================================================
    # SECCIÓN 7: PROTOCOLO DE EVALUACIÓN
    # ============================================================
    st.markdown("---")
    st.markdown("### 📊 Protocolo de Evaluación")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🔄 Validación Cruzada
        
        - **Método:** K-Fold (5 folds)
        - **Estratificación:** Por distrito
        - **Semilla:** random_state=42
        - **Métrica principal:** R² y MAE
        """)
    
    with col2:
        st.markdown("""
        #### 🎯 Optimización
            
        - **Framework:** Optuna
        - **Trials:** 30 por modelo
        - **Modelos:** RandomForest, XGBoost, SVR
        - **Búsqueda:** Hiperparámetros automática
        """)
    
    with col3:
        st.markdown("""
        #### 📏 Baseline
            
        - **Modelo:** Predicción del promedio
        - **R²:** 0 (por definición)
        - **MAE:** 7.0297 puntos
        - **Comparación:** Todos los modelos vs baseline
        """)
    
# ============================================================
    # SECCIÓN 8: ENLACES Y RECURSOS
    # ============================================================
    st.markdown("---")
    st.markdown("### 🔗 Enlaces y Recursos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📄 Documentación
        
        - [📊 Mapa de Pobreza INEI 2018](https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib1744/libro.pdf)
        - [📚 SatMAE Paper](https://arxiv.org/abs/2207.08051)
        - [📚 DINOv2 Paper](https://arxiv.org/abs/2304.07193)
        - [📚 ResNet50 Paper](https://arxiv.org/abs/1512.03385)
        """)
    
    with col2:
        st.markdown("""
        #### 💻 Repositorios
        
        - [🐙 GitHub - SatMAE](https://github.com/sustainlab-group/SatMAE)
        - [🐙 GitHub - DINOv2](https://github.com/facebookresearch/dinov2)
        - [🤗 HuggingFace - ViT](https://huggingface.co/google/vit-large-patch16-224)
        - [🛰️ Google Earth Engine](https://developers.google.com/earth-engine)
        
        #### 📂 Datos del Proyecto
        
        - [📁 Dataset de Distritos (CSV)](https://drive.google.com/file/d/1SNs8f4taodcHjGFr-Fc6YX26x9Q9L1Vr/view?usp=sharing)
        - [📁 Imágenes Satelitales (GeoTIFF)](https://drive.google.com/drive/folders/1NghUOKzOSAf_2H-v8wKR4J7o9_cKdcc2?usp=sharing)
        - [📁 Repositorio del Proyecto](https://github.com/Alexssbs/web_puno_benchmark)
        """)
    
    # ============================================================
    # SECCIÓN 9: CITACIÓN
    # ============================================================
    st.markdown("---")
    st.markdown("### 📖 Citación")
    
    st.markdown("""
    ```bibtex
    @misc{chura2025benchmarking,
        title={Benchmarking Foundation Models for Poverty Prediction in the Peruvian Andes},
        author={Chura Tuni, Alex Franco},
        year={2025},
        institution={Universidad Nacional Del Altiplano},
        note={Comparative study of DINOv2, SatMAE, and ResNet-50}
    }
    """)
# ============================================================
# PIE DE PÁGINA
# ============================================================
st.markdown("---")

st.markdown("""
<div style="text-align: center; font-size: 12px; color: #999;">
    <b>Benchmarking Foundation Models para Predicción de Pobreza</b><br>
    Datos: INEI - Mapa de Pobreza 2018 |
    Imágenes: Sentinel-2 |
    Desarrollado con Streamlit
</div>
""", unsafe_allow_html=True)