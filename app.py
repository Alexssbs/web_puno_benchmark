"""
Benchmarking Foundation Models para Predicción de Pobreza
Dashboard interactivo - Puno, Perú
"""

import streamlit as st

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Benchmark GFMs - Puno",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIDEBAR - NAVEGACIÓN
# ============================================================
st.sidebar.markdown("""
<div style="text-align: center;">
    <h3>🌍 Benchmark GFMs</h3>
    <p style="font-size: 14px; color: #666;">Andes Peruanos - Puno</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "📊 Navegación",
    ["🏠 Inicio", "📈 Resultados", "📝 Metodología"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 12px; color: #999;">
    <b>Modelos evaluados:</b><br>
    ResNet50 · DINOv2 · ViT · SatMAE
    <br><br>
    <b>Distritos:</b> 110
    <br>
    <b>Año:</b> 2018
</div>
""", unsafe_allow_html=True)

# ============================================================
# CARGA DE PÁGINAS
# ============================================================
if pagina == "🏠 Inicio":
    import pages._01_inicio as page
elif pagina == "📈 Resultados":
    import pages._02_resultados as page
else:
    import pages._03_metodologia as page

page.show()