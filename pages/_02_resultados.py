"""
Página de Resultados - Dashboard de Métricas
Benchmarking GFMs en Puno
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from PIL import Image
import os

# ============================================================
# CACHE - CARGAR DATOS
# ============================================================
@st.cache_data
def cargar_datos():
    """Carga todos los datos necesarios para la página de resultados"""
    
    # 1. Dataset de distritos
    df_distritos = pd.read_csv('data/distritos.csv')
    df_distritos['ubigeo'] = df_distritos['ubigeo'].astype(str).str.zfill(6)
    
    # 2. Comparación SatMAE (predicciones reales)
    df_satmae = pd.read_csv('data/comparacion_satmae.csv')
    df_satmae['ubigeo'] = df_satmae['ubigeo'].astype(str).str.zfill(6)
    
    # 3. Resultados benchmark (3 modelos × 3 regresores)
    df_benchmark = pd.read_csv('data/resultados_benchmark.csv')
    
    # 4. Mejores resultados por modelo (hardcodeados de salidas.md)
    mejores_modelos = pd.DataFrame([
        {"Modelo": "ResNet50", "R2": -0.0455, "MAE": 7.2292, 
         "Reduccion": "PCA(100)", "Regresor": "SVR", "Dimensiones": 2048},
        {"Modelo": "DINOv2", "R2": -0.0171, "MAE": 7.0448, 
         "Reduccion": "UMAP(50)", "Regresor": "SVR", "Dimensiones": 768},
        {"Modelo": "ViT", "R2": -0.2584, "MAE": 7.9204, 
         "Reduccion": "PCA(100)", "Regresor": "GradientBoosting", "Dimensiones": 1024},
        {"Modelo": "SatMAE", "R2": 0.0033, "MAE": 6.9675, 
         "Reduccion": "PCA(50)", "Regresor": "SVR", "Dimensiones": 768}
    ])
    
    # 5. Baseline
    baseline_mae = 7.0297
    
    return df_distritos, df_satmae, df_benchmark, mejores_modelos, baseline_mae


# ============================================================
@st.cache_data
def cargar_imagen_distrito(ubigeo, distrito):
    """Busca y carga la imagen JPG del distrito"""
    nombre_limpio = distrito.upper().replace(' ', '_').replace('(', '').replace(')', '').replace('á','A').replace('é','E').replace('í','I').replace('ó','O').replace('ú','U')
    nombre_archivo = f"Puno_2018_{ubigeo}_{nombre_limpio}.jpg"
    
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
    """Función principal de la página de resultados"""
    
    st.title("📊 Dashboard de Resultados")
    st.markdown("Análisis comparativo de los modelos evaluados en la región Puno")
    
    # --- CARGAR DATOS ---
    df_distritos, df_satmae, df_benchmark, mejores_modelos, baseline_mae = cargar_datos()
    
    # ============================================================
    # SECCIÓN 1: TABLA COMPARATIVA PRINCIPAL
    # ============================================================
    st.markdown("---")
    st.markdown("### 📋 Tabla Comparativa de Modelos")
    st.markdown("Mejores resultados obtenidos por cada modelo con su mejor configuración")
    
    # Formatear tabla para mostrar
    df_mejores_display = mejores_modelos.copy()
    df_mejores_display['R2'] = df_mejores_display['R2'].round(4)
    df_mejores_display['MAE'] = df_mejores_display['MAE'].round(4)
    
    # Colorear según rendimiento
    st.dataframe(
        df_mejores_display.style
        .background_gradient(subset=['R2'], cmap='RdYlGn_r', vmin=-0.3, vmax=0.01)
        .background_gradient(subset=['MAE'], cmap='RdBu_r', vmin=6.9, vmax=8.0),
        use_container_width=True,
        hide_index=True
    )
    
    # ============================================================
    # SECCIÓN 2: GRÁFICOS DE BARRAS - R² y MAE
    # ============================================================
    st.markdown("---")
    st.markdown("### 📊 Comparación de Métricas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico R²
        fig_r2 = px.bar(
            mejores_modelos,
            x='Modelo',
            y='R2',
            color='R2',
            color_continuous_scale='RdYlGn_r',
            title="Coeficiente de Determinación (R²)",
            text='R2',
            height=400
        )
        fig_r2.add_hline(y=0, line_dash="dash", line_color="gray", 
                         annotation_text="Baseline (R²=0)")
        fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_r2.update_layout(
            coloraxis_showscale=False,
            yaxis_title="R²",
            yaxis_range=[-0.3, 0.02]
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        # Gráfico MAE
        fig_mae = px.bar(
            mejores_modelos,
            x='Modelo',
            y='MAE',
            color='MAE',
            color_continuous_scale='RdBu_r',
            title="Error Absoluto Medio (MAE)",
            text='MAE',
            height=400
        )
        fig_mae.add_hline(y=baseline_mae, line_dash="dash", line_color="gray",
                         annotation_text=f"Baseline MAE={baseline_mae:.4f}")
        fig_mae.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_mae.update_layout(
            coloraxis_showscale=False,
            yaxis_title="MAE (puntos porcentuales)",
            yaxis_range=[6.8, 8.0]
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    
    # ============================================================
    # SECCIÓN 3: SCATTER PLOT - Predicción vs Real (SatMAE)
    # ============================================================
    st.markdown("---")
    st.markdown("### 📈 Predicción vs Real - SatMAE")
    st.markdown("Comparación de la pobreza predicha por SatMAE vs la pobreza real (INEI)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Crear scatter plot
        fig_scatter = px.scatter(
            df_satmae,
            x='pob_pct',
            y='predicho_satmae',
            hover_data=['distrito', 'ubigeo', 'error_abs'],
            title="SatMAE: Predicción vs Realidad",
            labels={
                'pob_pct': 'Pobreza Real (%)',
                'predicho_satmae': 'Pobreza Predicha (%)'
            },
            height=500,
            color='error_abs',
            color_continuous_scale='RdBu_r',
            trendline='ols'
        )
        
        # Línea de referencia (perfecto)
        fig_scatter.add_trace(
            go.Scatter(
                x=[12, 70],
                y=[12, 70],
                mode='lines',
                name='Predicción Perfecta',
                line=dict(color='black', dash='dash')
            )
        )
        
        # Mejorar layout
        fig_scatter.update_layout(
            xaxis_range=[10, 70],
            yaxis_range=[10, 70],
            coloraxis_showscale=False,
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Estadísticas del scatter
        st.markdown("#### 📊 Estadísticas")
        
        corr = df_satmae['pob_pct'].corr(df_satmae['predicho_satmae'])
        mae_total = df_satmae['error_abs'].mean()
        mae_std = df_satmae['error_abs'].std()
        
        st.metric(
            label="Correlación",
            value=f"{corr:.4f}",
            help="Correlación de Pearson entre pobreza real y predicha"
        )
        
        st.metric(
            label="MAE Global",
            value=f"{mae_total:.4f}",
            delta=f"±{mae_std:.4f}",
            help="Error Absoluto Medio promedio"
        )
        
        # Mejores y peores predicciones
        mejor_pred = df_satmae.loc[df_satmae['error_abs'].idxmin()]
        peor_pred = df_satmae.loc[df_satmae['error_abs'].idxmax()]
        
        st.markdown("#### ✅ Mejor Predicción")
        st.markdown(f"**{mejor_pred['distrito']}**")
        st.markdown(f"Real: {mejor_pred['pob_pct']:.1f}% → Pred: {mejor_pred['predicho_satmae']:.1f}%")
        st.markdown(f"Error: {mejor_pred['error_abs']:.4f} puntos")
        
        st.markdown("#### ❌ Peor Predicción")
        st.markdown(f"**{peor_pred['distrito']}**")
        st.markdown(f"Real: {peor_pred['pob_pct']:.1f}% → Pred: {peor_pred['predicho_satmae']:.1f}%")
        st.markdown(f"Error: {peor_pred['error_abs']:.4f} puntos")
    
    # ============================================================
    # SECCIÓN 4: MAPA DE POBREZA (Real vs Predicho)
    # ============================================================
    st.markdown("---")
    st.markdown("### 🗺️ Mapas de Pobreza: Real vs Predicho (SatMAE)")
    
    # Selección de capa
    capa = st.radio(
        "Selecciona la capa a visualizar:",
        ["Pobreza Real (INEI)", "Pobreza Predicha (SatMAE)", "Error Absoluto"],
        horizontal=True
    )
    
    # Crear mapa
    m = folium.Map(location=[-15.5, -70.0], zoom_start=8)
    
    # Unir datos
    df_mapa = df_distritos.merge(df_satmae[['ubigeo', 'predicho_satmae', 'error_abs']], 
                                 on='ubigeo', how='left')
    
    for _, row in df_mapa.iterrows():
        # Determinar valor y color
        if capa == "Pobreza Real (INEI)":
            valor = row['pob_pct']
            if valor >= 50:
                color = 'red'
            elif valor >= 30:
                color = 'orange'
            else:
                color = 'green'
            label = f"Real: {valor:.1f}%"
        elif capa == "Pobreza Predicha (SatMAE)":
            valor = row['predicho_satmae']
            if pd.isna(valor):
                continue
            if valor >= 50:
                color = 'red'
            elif valor >= 30:
                color = 'orange'
            else:
                color = 'green'
            label = f"Pred: {valor:.1f}%"
        else:  # Error Absoluto
            valor = row['error_abs']
            if pd.isna(valor):
                continue
            if valor >= 15:
                color = 'red'
            elif valor >= 8:
                color = 'orange'
            else:
                color = 'green'
            label = f"Error: {valor:.1f}%"
        
        popup_html = f"""
        <div style="font-family: Arial; font-size: 13px; min-width: 150px;">
            <b>{row['distrito']}</b><br>
            <hr style="margin: 5px 0;">
            🏷️ Ubigeo: <b>{row['ubigeo']}</b><br>
            📊 {label}<br>
            📌 {row['lat']:.4f}, {row['lon']:.4f}
        </div>
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['distrito']} - {label}"
        ).add_to(m)
    
    folium_static(m, width=900, height=500)
    
    # ============================================================
    # SECCIÓN 5: ANÁLISIS PCA Y SHAP
    # ============================================================
    st.markdown("---")
    st.markdown("### 🔍 Análisis de Componentes")
    
    tab1, tab2 = st.tabs(["📊 Correlación PCA vs Pobreza", "🎯 Importancia SHAP"])
    
    with tab1:
        st.markdown("Correlación de las primeras 10 componentes principales con la pobreza")
        
        # Datos de correlaciones (hardcodeados de salidas.md)
        pca_data = pd.DataFrame([
            {"Componente": "PC1", "Correlación": -0.0963, "Varianza": 0.5523},
            {"Componente": "PC2", "Correlación": -0.1324, "Varianza": 0.0714},
            {"Componente": "PC3", "Correlación": -0.0211, "Varianza": 0.0630},
            {"Componente": "PC4", "Correlación": -0.1364, "Varianza": 0.0569},
            {"Componente": "PC5", "Correlación": -0.0948, "Varianza": 0.0343},
            {"Componente": "PC6", "Correlación": 0.0644, "Varianza": 0.0247},
            {"Componente": "PC7", "Correlación": 0.0856, "Varianza": 0.0223},
            {"Componente": "PC8", "Correlación": -0.0722, "Varianza": 0.0178},
            {"Componente": "PC9", "Correlación": -0.1635, "Varianza": 0.0136},
            {"Componente": "PC10", "Correlación": 0.0690, "Varianza": 0.0125}
        ])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_pca = px.bar(
                pca_data,
                x='Componente',
                y='Correlación',
                color='Correlación',
                color_continuous_scale='RdBu_r',
                title="Correlación de PCs con Pobreza",
                text='Correlación',
                height=400
            )
            fig_pca.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_pca.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_pca.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_pca, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Varianza Explicada")
            fig_var = px.line(
                pca_data,
                x='Componente',
                y='Varianza',
                title="Varianza Explicada Acumulada",
                markers=True,
                height=400
            )
            fig_var.update_layout(yaxis_title="Varianza Explicada")
            st.plotly_chart(fig_var, use_container_width=True)
    
    with tab2:
        st.markdown("Importancia de las componentes según SHAP (SatMAE + PCA + SVR)")
        
        # Datos SHAP (hardcodeados de salidas.md)
        shap_data = pd.DataFrame([
            {"Componente": "PC1", "Importancia": 0.5526},
            {"Componente": "PC4", "Importancia": 0.3200},
            {"Componente": "PC2", "Importancia": 0.1934},
            {"Componente": "PC6", "Importancia": 0.1062},
            {"Componente": "PC5", "Importancia": 0.0800},
            {"Componente": "PC12", "Importancia": 0.0767},
            {"Componente": "PC8", "Importancia": 0.0749},
            {"Componente": "PC16", "Importancia": 0.0635},
            {"Componente": "PC3", "Importancia": 0.0630},
            {"Componente": "PC11", "Importancia": 0.0311}
        ])
        
        fig_shap = px.bar(
            shap_data,
            x='Importancia',
            y='Componente',
            orientation='h',
            color='Importancia',
            color_continuous_scale='Blues',
            title="Top 10 Componentes más Importantes (SHAP)",
            text='Importancia',
            height=400
        )
        fig_shap.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_shap.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Importancia SHAP",
            yaxis_title=""
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    
    # ============================================================
    # SECCIÓN 6: RESULTADOS POR REGRESOR
    # ============================================================
    st.markdown("---")
    st.markdown("### 📊 Rendimiento por Regresor")
    st.markdown("Comparación de R² para diferentes regresores en cada modelo")
    
    # Filtrar solo los 3 modelos principales
    df_regresores = df_benchmark[df_benchmark['Embedding'].isin(['ResNet50', 'DINOv2', 'ViT'])]
    
    fig_regresores = px.bar(
        df_regresores,
        x='Embedding',
        y='R2_mean',
        color='Modelo_ML',
        barmode='group',
        title="R² por Modelo y Regresor",
        labels={
            'Embedding': 'Modelo',
            'R2_mean': 'R²',
            'Modelo_ML': 'Regresor'
        },
        text='R2_mean',
        height=400
    )
    fig_regresores.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_regresores.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    st.plotly_chart(fig_regresores, use_container_width=True)
    
    # ============================================================
    # SECCIÓN 7: EXPLORADOR DE DISTRITOS
    # ============================================================
    st.markdown("---")
    st.markdown("### 🖼️ Explorador de Distritos")
    
    # Selector de distrito
    distritos_list = sorted(df_distritos['distrito'].unique())
    distrito_seleccionado = st.selectbox(
        "Selecciona un distrito para ver detalles:",
        distritos_list
    )
    
    if distrito_seleccionado:
        # Obtener datos del distrito
        row = df_distritos[df_distritos['distrito'] == distrito_seleccionado].iloc[0]
        ubigeo = row['ubigeo']
        
        # Obtener predicción SatMAE
        pred_row = df_satmae[df_satmae['ubigeo'] == ubigeo]
        pred = pred_row['predicho_satmae'].values[0] if not pred_row.empty else None
        error = pred_row['error_abs'].values[0] if not pred_row.empty else None
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("#### 📍 Información")
            st.markdown(f"**Distrito:** {row['distrito']}")
            st.markdown(f"**Ubigeo:** {row['ubigeo']}")
            st.markdown(f"**Latitud:** {row['lat']:.4f}")
            st.markdown(f"**Longitud:** {row['lon']:.4f}")
            
            st.markdown("#### 📊 Pobreza")
            st.markdown(f"**Real:** {row['pob_pct']:.1f}%")
            if pred:
                st.markdown(f"**Predicha (SatMAE):** {pred:.1f}%")
                st.markdown(f"**Error:** {error:.4f} puntos")
        
        with col2:
            st.markdown("#### 🖼️ Imagen Satellite")
            img = cargar_imagen_distrito(ubigeo, distrito_seleccionado)
            if img:
                st.image(img, caption=f"Sentinel-2 - {distrito_seleccionado}", use_container_width=True)
            else:
                st.warning("Imagen no disponible")
        
        with col3:
            # Mini gráfico de comparación
            if pred:
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=['Real', 'Predicho'],
                    y=[row['pob_pct'], pred],
                    text=[f"{row['pob_pct']:.1f}%", f"{pred:.1f}%"],
                    textposition='outside',
                    marker_color=['#2E86C1', '#F5B041']
                ))
                fig_comp.update_layout(
                    title=f"Comparación - {distrito_seleccionado}",
                    yaxis_title="Pobreza (%)",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Evaluación del error
                if error < 1:
                    st.success(f"✅ Excelente predicción (error < 1 punto)")
                elif error < 3:
                    st.info(f"👍 Buena predicción (error < 3 puntos)")
                elif error < 6:
                    st.warning(f"⚠️ Predicción regular (error < 6 puntos)")
                else:
                    st.error(f"❌ Predicción pobre (error > 6 puntos)")
    
    # ============================================================
    # SECCIÓN 8: RESUMEN FINAL
    # ============================================================
    st.markdown("---")
    st.markdown("### 📌 Resumen de Hallazgos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 4px solid #4CAF50;">
            <b>✅ Hallazgos Principales</b>
            <ul style="margin-top: 10px;">
                <li><b>SatMAE</b> es el mejor modelo (R²=0.0033, MAE=6.97)</li>
                <li><b>Mejora marginal</b> vs baseline del promedio (MAE 6.97 vs 7.03)</li>
                <li>Los modelos <b>RGB</b> no superan al baseline</li>
                <li>El uso de <b>10 bandas espectrales</b> ofrece ventaja</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 15px; border-radius: 10px; border-left: 4px solid #FF9800;">
            <b>⚠️ Limitaciones</b>
            <ul style="margin-top: 10px;">
                <li>Dataset pequeño: <b>110 distritos</b></li>
                <li>Buffer espacial de <b>500m</b> es limitado</li>
                <li>La pobreza es <b>multidimensional</b></li>
                <li>Se necesitan <b>datos complementarios</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================
    # PIE DE PÁGINA
    # ============================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 12px; color: #999;">
        <b>Benchmarking Foundation Models para Predicción de Pobreza</b><br>
        Datos: INEI - Mapa de Pobreza 2018 | Imágenes: Sentinel-2 | Desarrollado con Streamlit
    </div>
    """, unsafe_allow_html=True)