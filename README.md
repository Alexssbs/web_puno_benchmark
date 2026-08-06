Perfecto. Aquí tienes el **README.md** actualizado con la sección del pipeline.

---

## 📄 **README.md** (Actualizado)


# 🌍 Benchmarking Foundation Models para Predicción de Pobreza

<div align="center">
  <br>
  <p><strong>Análisis comparativo de GFMs en los Andes Peruanos - Puno 2018</strong></p>
  <p>
    <a href="#-demo">Demo</a> •
    <a href="#-estructura-del-proyecto">Estructura</a> •
    <a href="#-instalación">Instalación</a> •
    <a href="#-uso">Uso</a> •
    <a href="#-resultados">Resultados</a>
  </p>
</div>

---

## 📋 Resumen

Este proyecto presenta un **dashboard interactivo** para visualizar los resultados de un benchmarking de **Modelos Fundacionales Geoespaciales (GFMs)** aplicados a la predicción de pobreza monetaria en la región Puno, Perú.

Además, incluye el **pipeline completo de procesamiento** implementado en Google Colab, que cubre desde la descarga de imágenes satelitales hasta la extracción de embeddings y evaluación de modelos.

### 🎯 Objetivos

- Evaluar comparativamente 4 modelos: ResNet50, DINOv2, ViT y SatMAE
- Visualizar métricas de rendimiento (R², MAE) en un dashboard interactivo
- Explorar la relación entre embeddings satelitales y pobreza distrital
- Proveer una herramienta reproducible para futuras investigaciones
- Documentar el pipeline completo de procesamiento de datos

### 📊 Modelos Evaluados

| Modelo | Tipo | Dimensiones | Bandas | Mejor R² | Mejor MAE |
|--------|------|-------------|--------|----------|-----------|
| **SatMAE** | Masked Autoencoder | 768 | 10 | **0.0033** | **6.9675** |
| DINOv2 | ViT Self-Supervised | 768 | 3 | -0.0171 | 7.0448 |
| ResNet50 | CNN Supervisada | 2048 | 3 | -0.0455 | 7.2292 |
| ViT | Vision Transformer | 1024 | 3 | -0.2584 | 7.9204 |

> **Baseline (promedio):** MAE = 7.0297 | R² = 0

---

## 🚀 Demo

Puedes ver el dashboard en funcionamiento aquí:

🔗 **[https://benchmark-puno.streamlit.app](https://benchmark-puno.streamlit.app)**

---

## 📁 Estructura del Proyecto

```text
web_puno_benchmark/
│
├── pipeline.ipynb                  # Pipeline completo en Google Colab
├── app.py                          # Aplicación principal
├── requirements.txt                # Dependencias
│
├── pages/
│   ├── __init__.py
│   ├── _01_inicio.py
│   ├── _02_resultados.py
│   └── _03_metodologia.py
│
├── data/
│   ├── distritos.csv
│   ├── comparacion_satmae.csv
│   └── resultados_benchmark.csv
│
├── GEE_Puno_2018_JPG/
│   └── Puno_2018_*.jpg
│
├── assets/
│   ├── logo.png
│   └── style.css
│
└── README.md
```

---

## 📓 Pipeline (pipeline.ipynb)

El archivo **`pipeline.ipynb`** contiene el código completo utilizado para:

### 🔄 Fases del Pipeline

1. **Descarga de Imágenes**
   - Conexión con Google Earth Engine
   - Filtro de nubes (< 20%)
   - Composite mediano anual (2018)
   - Exportación a Google Drive

2. **Pre-procesamiento**
   - Limpieza de valores NaN e Inf
   - Normalización de reflectancia
   - Generación de miniaturas JPG

3. **Extracción de Embeddings**
   - ResNet50 (2048 dimensiones)
   - DINOv2 (768 dimensiones)
   - ViT (1024 dimensiones)
   - SatMAE (768 dimensiones, 10 bandas)

4. **Evaluación de Modelos**
   - Optimización con Optuna
   - Validación cruzada (5 folds)
   - Comparación de regresores
   - Análisis SHAP

### 🛠️ Tecnologías Utilizadas

| Herramienta | Uso |
|-------------|-----|
| **Google Earth Engine** | Descarga de imágenes Sentinel-2 |
| **PyTorch** | Extracción de embeddings |
| **Scikit-learn** | Regresión y evaluación |
| **Optuna** | Optimización de hiperparámetros |
| **UMAP** | Reducción dimensional |
| **SHAP** | Interpretabilidad de modelos |

> **Nota:** El pipeline está diseñado para ejecutarse en Google Colab con acceso a GPU.

---

## 📦 Datos

### 📊 Dataset de Distritos
- **Archivo:** `data/distritos.csv`
- **Contenido:** 110 distritos de Puno con coordenadas y pobreza real
- **Fuente:** INEI - Mapa de Pobreza Monetaria 2018

### 🛰️ Imágenes Satelitales
- **Formato:** JPG (miniaturas para visualización)
- **Tamaño:** 400x400 píxeles
- **Fuente:** Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED)
- **Original:** GeoTIFF (12 bandas, 10m resolución)

### 📈 Resultados de Benchmarking
- **Archivo:** `data/resultados_benchmark.csv`
- **Contenido:** Métricas de 3 modelos × 3 regresores

---

## 🔧 Instalación

### Requisitos previos

- Python 3.9+
- pip

### Pasos


# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/benchmark-puno.git
cd benchmark-puno

# 2. Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py


La aplicación estará disponible en: `http://localhost:8501`

---

## 🖥️ Uso

### Páginas disponibles

1. **🏠 Inicio**
   - Mapa interactivo de Puno con distritos coloreados por pobreza
   - Estadísticas clave del estudio
   - Vista previa de resultados principales

2. **📈 Resultados**
   - Tabla comparativa de modelos
   - Gráficos de R² y MAE
   - Scatter plot: Predicción vs Real (SatMAE)
   - Mapas de pobreza (Real, Predicho, Error)
   - Análisis PCA y SHAP
   - Explorador de distritos con imágenes satelitales

3. **📝 Metodología**
   - Diagrama del pipeline
   - Descripción detallada de modelos
   - Técnicas de reducción dimensional
   - Regresores evaluados
   - Protocolo de evaluación

---

## 📊 Resultados Principales

### 🏆 Mejor Modelo: SatMAE

| Métrica | Valor |
|---------|-------|
| **R²** | 0.0033 |
| **MAE** | 6.9675 puntos porcentuales |
| **Mejor Reducción** | PCA (50 componentes) |
| **Mejor Regresor** | SVR (RBF) |
| **Mejora vs Baseline** | 0.0622 puntos en MAE |

### 📈 Comparación de Modelos


R²:  SatMAE (0.0033) > DINOv2 (-0.0171) > ResNet50 (-0.0455) > ViT (-0.2584)
MAE: SatMAE (6.97)  < DINOv2 (7.04)   < ResNet50 (7.23)   < ViT (7.92)


---

## 🔗 Enlaces y Recursos

### 📄 Documentación

- [📊 Mapa de Pobreza INEI 2018](https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib1744/libro.pdf)
- [📚 SatMAE Paper](https://arxiv.org/abs/2207.08051)
- [📚 DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [📚 ResNet50 Paper](https://arxiv.org/abs/1512.03385)

### 💻 Repositorios

- [🐙 GitHub - SatMAE](https://github.com/sustainlab-group/SatMAE)
- [🐙 GitHub - DINOv2](https://github.com/facebookresearch/dinov2)
- [🤗 HuggingFace - ViT](https://huggingface.co/google/vit-large-patch16-224)
- [🛰️ Google Earth Engine](https://developers.google.com/earth-engine)

### 📂 Datos del Proyecto

Los datos utilizados en este proyecto están disponibles en Google Drive:

- [📁 Imágenes Satelitales (GeoTIFF)](https://drive.google.com/drive/folders/1NghUOKzOSAf_2H-v8wKR4J7o9_cKdcc2?usp=sharing)
- [📁 Miniaturas JPG (Ubicado en la carpeta data/)
- [📁 Dataset de Distritos (CSV)](https://drive.google.com/file/d/1SNs8f4taodcHjGFr-Fc6YX26x9Q9L1Vr/view?usp=sharing)

> **Nota:** Las imágenes GeoTIFF son archivos grandes (~50 MB cada uno). Las miniaturas JPG están optimizadas para visualización en el dashboard.

---

## 🔄 Reproducibilidad

Para reproducir completamente este estudio:

### 1. Ejecutar el Pipeline
Abre `pipeline.ipynb` en Google Colab y ejecuta todas las celdas.

### 2. Generar Miniaturas
El pipeline incluye código para generar miniaturas JPG de las imágenes GeoTIFF.

### 3. Extraer Embeddings
El pipeline extrae embeddings de los 4 modelos evaluados.

### 4. Evaluar Modelos
Ejecuta las celdas de evaluación para obtener las métricas reportadas.

---

# 👨‍💻 Autor

**Alex Chura**

---
<div align="center">

### 🌍 Benchmarking Foundation Models para Predicción de Pobreza

*Desarrollado  utilizando Streamlit, PyTorch y Google Earth Engine.*

</div>

