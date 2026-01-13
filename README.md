# Dual_Pathway_EEG_Analysis
Dual-pathway EEG (Electroencephalogram) analysis approach that uses interpretable Machine Learning (ML) for micro-temporal attention profiling and Hybrid Deep Learning for reliable ADHD classification: (1) A CNN-Transformer-BiLSTM architecture (95% accuracy, AUC=0.989); and (2) XGBoost (F1-macro=83.10%, κ = 0.935) for classification profiles. Leveraging the strengths of DL and ML to complement their own weaknesses, it is
proposed, a dual-pathway architecture for ADHD classification and micro-temporal attention profile detection, integrating Hybrid DL with interpretable feature engineering and ML in EEG analysis to develop profile-based educational personalization.

# ============================================================================
# NOTAS DE INSTALACIÓN
# ============================================================================
### Requisitos Previos

- Python 3.8 a 3.11
- CUDA (opcional, para GPU)
- 8GB RAM mínimo

### Paso 1: Descargar

# descargar y extraer el ZIP
# descargar modelos: https://drive.google.com/drive/folders/1WTndymEbd0FAsJw9NiUI0lTkDJSxouzv?usp=drive_link
# copiar el folder principal (EEG_ADHD_v5) al proyecto
# copiar los modelos dentro del folder models a el folder principal EEG_ADHD_v5

# 1. Crear entorno virtual (recomendado):
#    python -m venv venv
#    
#    Windows: venv\Scripts\activate
#    Linux/Mac: source venv/bin/activate

# 2. Instalar dependencias:
#    pip install -r requirements.txt

# 3. Si usas GPU, instalar PyTorch manualmente:
#    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# Archivo de Dependencias
# Instalar con: pip install -r requirements.txt

# ============================================================================
# DEPENDENCIAS CORE
# ============================================================================

# Computación numérica
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0

# Archivos y datos
h5py>=3.6.0
joblib>=1.1.0

# Visualización
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilidades
tqdm>=4.62.0
pathlib>=1.0.1

# ============================================================================
# DEEP LEARNING (Para clasificación ADHD)
# ============================================================================

# PyTorch - Elegir UNA de las siguientes opciones:

# Opción 1: CPU only (más ligero)
# torch>=2.0.0
# torchvision>=0.15.0
# torchaudio>=2.0.0

# Opción 2: GPU CUDA 11.8 (mejor rendimiento)
# Instalar manualmente con:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Opción 3: GPU CUDA 12.1 (más reciente)
# Instalar manualmente con:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ============================================================================
# OPCIONALES (Mejoran funcionalidad pero no son críticas)
# ============================================================================

# Procesamiento de señales avanzado
# pywavelets>=1.3.0

# Aceleración de cálculos
# numba>=0.55.0

# Notebooks (si quieres ejecutar análisis exploratorios)
# jupyter>=1.0.0
# ipykernel>=6.0.0

# Métricas adicionales
# imbalanced-learn>=0.9.0

# ============================================================================
# VERSIONES PROBADAS
# ============================================================================
# Python: 3.8.10, 3.9.13, 3.11
# OS: Windows 10/11, Ubuntu 20.04/22.04, macOS 12+
# RAM: Mínimo 8GB, Recomendado 16GB
# GPU: Opcional, mejora velocidad de ADHD training


*****************USO DE LOS MODULOS********************************************
# =============================================================================
# VERIFICAR FORMATO ARCHIVO A PROCESAR
# =============================================================================

# FORMATO ARCHIVO PARA PROCESAR DEBE SER csv CON LA SIGUIENTE ESTRUCTURA:

Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fz,Cz,Pz,Class,ID
261.0,402.0,16.0,261.0,126.0,384.0,126.0,236.0,52.0,236.0,200.0,16.0,200.0,494.0,126.0,236.0,121.0,367.0,121.0,,v10p

# LA COLUMNA Class EN BLANCO QUE ES LA QUE SE VA A PREDECIR (ADHD,Control)

# ===============================================================================
# FASE 1: ANALISIS EXPLORATORIO DE LOS DATOS, DIVISION MULTIESCALA DE TIEMPO, PREPROCESAMIENTO Y    # EXTRACCION DE CARACTERISTICAS TEMPORALES
# ===============================================================================

1. Copiamos el archivo de señales EEG a utilizar a la ruta '../EEG_ADHD_v5/data/raw/' dentro del proyecto.
2. Cambiamos el nombre del archivo a adhdata.csv
3. Ejecutamos Fase1.py

SALIDA GENERADA:

* Datos preprocesados listos para aplicar los modelos DL y ML
* Reporte de calidad de las señales, visualizaciones

# ================================================================================
# FASE 2: APLICAR MODELO DL HIBRIDO PARA CLASIFICACION ADHD
# ================================================================================
	
1. Ejecutamos model_evaluator_v2.py

SALIDA GENERADA:

* Evaluación
    RESUMEN DE RESULTADOS:
     
    Accuracy: 
    ROC-AUC:
    F1-Score: 
    Nivel Sujeto:
    Accuracy: 
    ROC-AUC: 
    F1-Score: 
    Confianza: 
    Archivos generados: '../EEG_ADHD_v5/results/evaluation/'
    adhd_predictions.csv 
    subject_predictions_detailed.csv
    evaluation_metrics.json
    Visualizaciones en plots/

# ==================================================================================
# FASE 3: MODELO ML PARA CLASIFICACION DE PERFILES ATENCIONALES 
# ==================================================================================

1. Ejecutar Fase3.py

# ==================================================================================
# FASE 4: INTEGRACION DIAGNOSTICO ADHD + PERFIL (PERFIL COMPLETO)
# =================================================================================

1. Ejecutar Fase4.py

SALIDA GENERADA:

* PERFILES:
    Archivos generados: '../EEG_ADHD_v5/profiles/'
    Visualizaciones
# ================================================================================
# FASE 5: GENERAR PLANES DE INTERVENCION EDUCATIVA PERSONALIZADOS
# ================================================================================    

1. Ejecutar Intervention_planner_examples.py

SALIDA GENERADA:
* PLANES DE INTERVENCION EDUCATIVA PERSONALIZADA:
    Archivos generados: '../EEG_ADHD_v5/intervention_plans/'
    	
# ================================================================================
# Adicional: MAPAS DE ATENCION
# ================================================================================

1. Ejecutar	attention_mapper_examples.py

SALIDA GENERADA:
* MAPAS DE ANTENCION:
    Archivos generados: '../EEG_ADHD_v5/attention_maps/'
