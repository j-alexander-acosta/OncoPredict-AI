# OncoPredict AI - Detección y Evaluación de Riesgo de Cáncer Cervicouterino

OncoPredict AI es una aplicación web impulsada por Machine Learning (Inteligencia Artificial) diseñada para la detección temprana y evaluación del riesgo del cáncer cervicouterino. 

Este proyecto utiliza una arquitectura de procesamiento triple que combina: 
1. Modelos predictivos tradicionales basados en factores de riesgo clínicos (Datos Tabulares).
2. Visión por Computadora mediante Redes Neuronales Convolucionales (CNN) para el análisis microscópico de citologías.
3. Visión por Computadora para el análisis macroscópico de tejido cervical in-vivo (Colposcopía).

## Características Principales

1. **Inferencia Clínica Individual (Datos Tabulares)**
   - Utiliza datos de factores de riesgo demográficos, de estilo de vida e historial médico de la paciente.
   - Entrenado con el dataset de cáncer cervicouterino del **UCI Machine Learning Repository** (858 pacientes).
   - Implementa imputación de datos faltantes, escalado y balanceo de clases sintético (SMOTE).
   - **Modelos Integrados:** Regresión Logística, Support Vector Machine (SVM), Árbol de Decisión, Random Forest y XGBoost.
   - Entrega un dictamen de riesgo (Bajo, Moderado, Alto) y provee un consenso entre todos los modelos para mayor seguridad diagnóstica.

2. **Benchmark Comparativo de Modelos**
   - Panel de rendimiento en tiempo real que compara las métricas clave (Accuracy, Sensibilidad, Especificidad, Precisión, F1-Score, AUC-ROC) de los algoritmos de clasificación.
   - Aparta claramente las evaluaciones de modelos de **datos tabulares**, de **imágenes microscópicas (Citología)** y de **imágenes macroscópicas (Colposcopía)**, permitiendo comparar el desempeño entre datasets.
   - Ayuda a los profesionales de la salud y científicos de datos a entender qué modelo es más apto según el umbral de falsos positivos/negativos tolerado en cribados.

3. **Análisis de Imagen de Citología (Microscopía / Visión por Computadora)**
   - Utiliza *Transfer Learning* y **Fine-Tuning Progresivo** sobre arquitecturas profundas probadas para detectar células anormales en muestras celulares de microscopio.
   - **Datasets Soportados:** **Herlev** (917 imágenes), **SIPaKMeD** (~4049 imágenes) y **RIVA**, distribuidas en diversas clases morfológicas.
   - Agrupa los hallazgos en "Bajo Riesgo" (ej. normal_columnar, superficial) o "Alto Riesgo" (ej. carcinoma_in_situ, dysplastic).
   - **Modelos Integrados:** **EfficientNet-B0 (Recomendado/SOTA)**, MobileNet, InceptionV3, ResNet50 y AlexNet. EfficientNet alcanzó un 94.4% de precisión solucionando el estancamiento morfológico de las redes previas.
   - **Sistema de Explicabilidad (XAI):** Integra **Score-CAM** (mapas de alta resolución sin gradientes), evaluación local **Deletion AUC** (~0.2890) y verificación global **TCAV** (Sensibilidad conceptual 100%), eliminando el efecto "caja negra" y elevando la auditabilidad a grado médico.
   - Cuenta con soporte de aceleración gráfica **MPS (Apple Silicon)** para entrenamientos locales en Mac.

4. **Análisis de Colposcopía (Macroscopía In-Vivo / Visión por Computadora)**
   - Evalúa imágenes macroscópicas del cuello uterino capturadas durante la colposcopía para identificar y clasificar el tipo anatómico de la lesión.
   - **Dataset Soportado:** **Intel & MobileODT Cervical Cancer Screening** (Kaggle), que clasifica los cuellos uterinos según el tipo de Zona de Transformación.
   - Clasifica los hallazgos en 3 categorías médicas:
     - **Tipo 1:** Unión escamocolumnar completamente visible (Bajo riesgo).
     - **Tipo 2:** Unión parcialmente visible (Riesgo moderado).
     - **Tipo 3:** Unión no visible, ubicada en el canal endocervical (Alto riesgo de lesiones ocultas).
   - **Modelo Integrado:** **EfficientNet-B0**, adaptado mediante fine-tuning progresivo y modificado en su última capa para inferir entre las 3 clases anatómicas.

## Datasets Utilizados

Para el desarrollo y entrenamiento de los distintos modelos, este proyecto emplea los siguientes conjuntos de datos de acceso público y rigor académico:

- **[Cervical cancer (Risk Factors)](https://www.nature.com/articles/s41597-025-06280-2)** (Vía UCI Machine Learning / Nature): Conjunto de datos tabulares con variables demográficas, hábitos y factores de riesgo clínicos.
- **[Herlev Dataset](https://www.kaggle.com/datasets/ayaanelahi/herlev-cervical-cancer-dataset)**: Dataset de imágenes de citologías microscópicas recolectadas en el Hospital Universitario Herlev.
- **[SIPaKMeD Dataset](https://www.cs.uoi.gr/~marina/sipakmed.html)**: Base de datos exhaustiva de imágenes de células cervicales segmentadas para el análisis morfológico.
- **[RIVA Dataset](https://www.nature.com/articles/s41597-025-06280-2)**: Dataset reciente de citologías empleado para robustecer el análisis de anomalías microscópicas.
- **[Intel & MobileODT Cervical Cancer Screening](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening)**: Competición de Kaggle con imágenes macroscópicas in-vivo (colposcopías) para clasificar la Zona de Transformación del cuello uterino.

## Tecnologías Utilizadas

- **Backend:** Python 3, Flask.
- **Machine Learning (Tabular):** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE).
- **Visión por Computadora (Imágenes):** PyTorch (para EfficientNet-B0/AlexNet), TensorFlow/Keras (para MobileNet/Inception), Pillow (PIL).
- **Frontend:** HTML5, Vanilla JavaScript, CSS3 (Diseño responsivo y adaptativo).
- **Manipulación de Datos:** Pandas, Numpy.

## Estructura del Directorio

```text
Análisis Comparativo/
├── app.py                     # Archivo principal del servidor Flask
├── analisis_comparativo.py    # Script ETL y pipeline de entrenamiento base
├── model_service.py           # Servicio que carga los modelos pre-entrenados y orquesta las predicciones
├── requirements.txt           # Dependencias del entorno Python
├── Media/
│   ├── risk_factors_cervical_cancer.csv  # Dataset de factores de riesgo clínicos (UCI)
│   ├── Herlev Dataset/                   # Dataset de imágenes microscópicas de citologías
│   ├── SIPaKMeD/                         # Dataset de citologías microscópicas
│   ├── RIVA/                             # Dataset de citologías microscópicas
│   ├── intel-mobileodt-cervical-cancer-screening/ # Dataset de imágenes macroscópicas de colposcopía
│   └── models/                           # Directorio donde se guardan los modelos pre-entrenados (.keras y .pth)
├── static/
│   ├── css/
│   │   └── style.css          # Estilos de la UI
│   └── js/
│       └── main.js            # Lógica de la interfaz gráfica e integraciones AJAX
└── templates/
    └── index.html             # Plantilla HTML principal
```

## Pasos para la Implementación (Instalación y Ejecución)

Sigue estos pasos para implementar y ejecutar el proyecto de forma local:

### 1. Clonar y Configurar Entorno Virtual
Asegúrate de tener Python 3 instalado. Abre una terminal en la raíz del proyecto y crea un entorno virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # En Linux/Mac
# .venv\Scripts\activate   # En Windows
```

### 2. Instalar Dependencias
Instala los paquetes necesarios utilizando el archivo de requisitos:
```bash
pip install -r requirements.txt
```
*(Si `requirements.txt` no tiene TensorFlow, puedes instalarlo manualmente con: `pip install tensorflow pillow`)*

### 3. Preparar los Datos (Media)
Asegúrate de que los datasets estén en sus rutas correctas dentro de la carpeta `Media/`:
- El archivo `risk_factors_cervical_cancer.csv`.
- La carpeta `Herlev Dataset` con las subcarpetas `train` y `test`, y dentro de ellas las clases morfológicas.

### 4. Entrenar y Generar los Modelos
El servicio web requiere que los modelos existan en la carpeta `Media/models/`. Para generarlos, debes ejecutar los scripts de entrenamiento:
```bash
# Entrenamiento tradicional (Modelos base)
python analisis_comparativo.py

# Entrenamiento SOTA para Citologías Microscópicas (PyTorch/MPS)
python train_missing.py

# Entrenamiento SOTA para Colposcopías Macroscópicas (PyTorch/MPS)
python train_colposcopy.py
```
> *Nota: Este proceso entrenará las redes neuronales y el modelo tabular. Puede demorar varios minutos dependiendo de los recursos del equipo.*

### 5. Iniciar la Aplicación Web
Una vez entrenados los modelos, levanta el servidor Flask:
```bash
python app.py
```
El servidor se iniciará en `http://127.0.0.1:5001`. 

### 6. Uso
1. Abre tu navegador e ingresa a `http://127.0.0.1:5001/`.
2. Para **Datos Clínicos**, llena el formulario en la primera pestaña y haz clic en "Ejecutar Análisis".
3. Para **Imágenes**, ve a la pestaña de "Análisis de Imagen", selecciona un modelo (por ejemplo, MobileNet), sube una imagen citológica y obtén tu dictamen en segundos.
