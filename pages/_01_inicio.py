"""
Página de Inicio - Benchmarking GFMs en Puno
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
import os
from PIL import Image
import numpy as np

# ============================================================
# CACHE - CARGAR DATOS
# ============================================================
@st.cache_data
def cargar_datos():
    """Carga los datos necesarios para la página de inicio"""
    
    # 1. Dataset de distritos
    df = pd.read_csv('data/distritos.csv')
    df['ubigeo'] = df['ubigeo'].astype(str).str.zfill(6)
    
    # 2. Resultados comparativos (hardcodeados de salidas.md)
    resultados = [
        {"Modelo": "ResNet50", "R2": -0.0455, "MAE": 7.2292, 
         "Reduccion": "PCA(100)", "Regresor": "SVR"},
        {"Modelo": "DINOv2", "R2": -0.0171, "MAE": 7.0448, 
         "Reduccion": "UMAP(50)", "Regresor": "SVR"},
        {"Modelo": "ViT", "R2": -0.2584, "MAE": 7.9204, 
         "Reduccion": "PCA(100)", "Regresor": "GradientBoosting"},
        {"Modelo": "SatMAE", "R2": 0.0033, "MAE": 6.9675, 
         "Reduccion": "PCA(50)", "Regresor": "SVR"}
    ]
    df_resultados = pd.DataFrame(resultados)
    
    # 3. Baseline (promedio)
    baseline_mae = 7.0297
    
    return df, df_resultados, baseline_mae


# ============================================================
@st.cache_data
def cargar_imagen_distrito(ubigeo, distrito):
    """Busca y carga la imagen JPG del distrito"""
    
    # Limpiar nombre para buscar archivo
    nombre_limpio = distrito.upper().replace(' ', '_').replace('(', '').replace(')', '')
    nombre_archivo = f"Puno_2018_{ubigeo}_{nombre_limpio}.jpg"
    
    # Buscar en la carpeta de imágenes
    carpeta_imagenes = 'GEE_Puno_2018_JPG'
    ruta = os.path.join(carpeta_imagenes, nombre_archivo)
    
    if os.path.exists(ruta):
        try:
            img = Image.open(ruta)
            return img
        except:
            return None
    return None


# ============================================================
def show():
    """Función principal de la página de inicio"""
    
    # --- TÍTULO ---
    st.title("🌍 Benchmarking Foundation Models")
    st.subheader("Predicción de Pobreza en los Andes Peruanos - Puno 2018")
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        Este dashboard presenta los resultados de un estudio comparativo de 
        <b>Modelos Fundacionales Geoespaciales (GFMs)</b> para la estimación de 
        pobreza monetaria a nivel distrital en la región Puno, Perú.
    </div>
    """, unsafe_allow_html=True)
    
    # --- CARGAR DATOS ---
    df, df_resultados, baseline_mae = cargar_datos()
    
    # --- ESTADÍSTICAS CLAVE ---
    st.markdown("### 📊 Estadísticas del Estudio")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📋 Distritos",
            value=f"{len(df)}",
            help="Total de distritos de Puno incluidos en el estudio"
        )
    
    with col2:
        st.metric(
            label="🧠 Modelos Evaluados",
            value="4",
            help="ResNet50, DINOv2, ViT, SatMAE"
        )
    
    with col3:
        mejor_r2 = df_resultados['R2'].max()
        mejor_modelo = df_resultados[df_resultados['R2'] == mejor_r2]['Modelo'].values[0]
        st.metric(
            label=f"📈 Mejor R² ({mejor_modelo})",
            value=f"{mejor_r2:.4f}",
            delta="vs Baseline (0)",
            help="SatMAE logró el mejor R², superando ligeramente al baseline"
        )
    
    with col4:
        mejor_mae = df_resultados['MAE'].min()
        mejor_modelo_mae = df_resultados[df_resultados['MAE'] == mejor_mae]['Modelo'].values[0]
        mejora = baseline_mae - mejor_mae
        st.metric(
            label=f"📉 Mejor MAE ({mejor_modelo_mae})",
            value=f"{mejor_mae:.4f}",
            delta=f"-{mejora:.4f} vs baseline",
            help=f"Baseline (promedio): MAE = {baseline_mae:.4f}"
        )
    
    # --- MÁS ESTADÍSTICAS (fila 2) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rango_pobreza = f"{df['pob_pct'].min():.1f}% - {df['pob_pct'].max():.1f}%"
        st.metric(
            label="📊 Rango de Pobreza",
            value=rango_pobreza,
            help="Pobreza monetaria (% de población en pobreza)"
        )
    
    with col2:
        st.metric(
            label="📅 Año de Referencia",
            value="2018",
            help="Alineado con el Mapa de Pobreza del INEI"
        )
    
    with col3:
        st.metric(
            label="🛰️ Imágenes",
            value="Sentinel-2",
            help="12 bandas, resolución 10m, composite mediano anual"
        )
    
    st.markdown("---")
    
    # --- MAPA DE PUNO ---
    st.markdown("### 🗺️ Mapa de Pobreza por Distrito")
    st.markdown("""
    <div style="font-size: 14px; color: #666; margin-bottom: 10px;">
        🔴 <b>Alta pobreza</b> (&gt;50%) &nbsp;|&nbsp; 
        🟠 <b>Media pobreza</b> (30-50%) &nbsp;|&nbsp; 
        🟢 <b>Baja pobreza</b> (&lt;30%)
    </div>
    """, unsafe_allow_html=True)
    
    # Crear mapa con Folium
    m = folium.Map(
        location=[-15.5, -70.0],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Añadir marcadores por distrito
    for _, row in df.iterrows():
        # Color según nivel de pobreza
        if row['pob_pct'] >= 50:
            color = 'red'
            fill_color = 'red'
        elif row['pob_pct'] >= 30:
            color = 'orange'
            fill_color = 'orange'
        else:
            color = 'green'
            fill_color = 'green'
        
        # Popup con información
        popup_html = f"""
        <div style="font-family: Arial; font-size: 13px; min-width: 150px;">
            <b>{row['distrito']}</b><br>
            <hr style="margin: 5px 0;">
            🏷️ Ubigeo: <b>{row['ubigeo']}</b><br>
            📊 Pobreza: <b>{row['pob_pct']:.1f}%</b><br>
            📌 {row['lat']:.4f}, {row['lon']:.4f}
        </div>
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.7,
            weight=2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['distrito']} - {row['pob_pct']:.1f}%"
        ).add_to(m)
    
    # Mostrar mapa
    folium_static(m, width=900, height=500)
    
    # --- TABLA DE DISTRITOS (resumida) ---
    with st.expander("📋 Ver lista completa de distritos"):
        st.dataframe(
            df[['ubigeo', 'distrito', 'pob_pct', 'lat', 'lon']]
            .sort_values('pob_pct', ascending=False)
            .style.background_gradient(subset=['pob_pct'], cmap='RdYlGn_r'),
            use_container_width=True,
            height=400
        )
    
    st.markdown("---")
    
    # --- RESULTADOS PRINCIPALES (vista previa) ---
    st.markdown("### 🏆 Resultados Principales")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Gráfico de barras - R²
        fig_r2 = px.bar(
            df_resultados,
            x='Modelo',
            y='R2',
            color='R2',
            color_continuous_scale='RdYlGn_r',
            title="Coeficiente de Determinación (R²)",
            text='R2',
            height=350
        )
        fig_r2.add_hline(y=0, line_dash="dash", line_color="gray", 
                         annotation_text="Baseline (R²=0)")
        fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_r2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        # Gráfico de barras - MAE
        fig_mae = px.bar(
            df_resultados,
            x='Modelo',
            y='MAE',
            color='MAE',
            color_continuous_scale='RdBu_r',
            title="Error Absoluto Medio (MAE)",
            text='MAE',
            height=350
        )
        # Línea del baseline
        fig_mae.add_hline(y=baseline_mae, line_dash="dash", line_color="gray",
                         annotation_text=f"Baseline MAE={baseline_mae:.4f}")
        fig_mae.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_mae.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_mae, use_container_width=True)
    
    # --- DESCRIPCIÓN DEL MEJOR MODELO ---
    st.markdown("---")
    st.markdown("### 🎯 Mejor Modelo: SatMAE")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 4px solid #4CAF50;">
            <b>🔬 Modelo</b><br>
            SatMAE (Masked Autoencoder)<br>
            <span style="font-size: 12px; color: #666;">Pre-entrenado en fMoW-Temporal</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 4px solid #2196F3;">
            <b>📊 Métricas</b><br>
            R² = <b>0.0033</b><br>
            MAE = <b>6.97</b> puntos porcentuales
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color: #fff3e0; padding: 15px; border-radius: 10px; border-left: 4px solid #FF9800;">
            <b>📈 Mejora vs Baseline</b><br>
            MAE: {baseline_mae:.4f} → {df_resultados[df_resultados['Modelo']=='SatMAE']['MAE'].values[0]:.4f}<br>
            <span style="color: #4CAF50;">⬆ {baseline_mae - df_resultados[df_resultados['Modelo']=='SatMAE']['MAE'].values[0]:.4f} puntos</span>
        </div>
        """, unsafe_allow_html=True)
    
    # --- BOTONES DE NAVEGACIÓN ---
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Ver Dashboard de Resultados", use_container_width=True):
            st.switch_page("app.py")
            # Nota: en Streamlit, para cambiar de página se usa st.switch_page
            # Pero como estamos en app.py, mejor redirigir con query params
    
    with col2:
        if st.button("📝 Ver Metodología", use_container_width=True):
            st.switch_page("app.py")
    
    with col3:
        st.markdown("""
        <div style="text-align: center; font-size: 12px; color: #999; padding-top: 10px;">
            📄 <a href="#" style="color: #999;">Descargar artículo completo (PDF)</a>
        </div>
        """, unsafe_allow_html=True)
    
    # --- PIE DE PÁGINA ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 12px; color: #999;">
        <b>Benchmarking Foundation Models para Predicción de Pobreza</b><br>
        Datos: INEI - Mapa de Pobreza 2018 | Imágenes: Sentinel-2 | Desarrollado con Streamlit
    </div>
    """, unsafe_allow_html=True)