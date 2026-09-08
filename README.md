# OncoPredict AI - Detección y Evaluación de Riesgo de Cáncer Cervicouterino

OncoPredict AI es una aplicación web impulsada por Machine Learning (Inteligencia Artificial) diseñada para la detección temprana y evaluación del riesgo del cáncer cervicouterino. 

Este proyecto utiliza una arquitectura de procesamiento dual (Fase 1 y Fase 2) que combina modelos predictivos tradicionales basados en factores de riesgo clínicos, con visión por computadora mediante Redes Neuronales Convolucionales (CNN) para el análisis de citologías.

## Características Principales

1. **Inferencia Clínica Individual (Datos Tabulares)**
   - Utiliza datos de factores de riesgo demográficos, de estilo de vida e historial médico de la paciente.
   - Entrenado con el dataset de cáncer cervicouterino del **UCI Machine Learning Repository** (858 pacientes).
   - Implementa imputación de datos faltantes, escalado y balanceo de clases sintético (SMOTE).
   - **Modelos Integrados:** Regresión Logística, Support Vector Machine (SVM), Árbol de Decisión, Random Forest y XGBoost.
   - Entrega un dictamen de riesgo (Bajo, Moderado, Alto) y provee un consenso entre todos los modelos para mayor seguridad diagnóstica.

2. **Benchmark Comparativo de Modelos**
   - Panel de rendimiento en tiempo real que compara las métricas clave (Accuracy, Sensibilidad, Especificidad, Precisión, F1-Score, AUC-ROC) de los algoritmos de clasificación de ambas fases.
   - Ayuda a los profesionales de la salud y científicos de datos a entender qué modelo es más apto según el umbral de falsos positivos/negativos tolerado en cribados.

3. **Análisis de Imagen de Citología (Visión por Computadora)**
   - Utiliza *Transfer Learning* sobre arquitecturas profundas probadas para detectar células anormales en muestras de citología (Dataset **Herlev** de 917 imágenes en 7 clases morfológicas).
   - Agrupa los hallazgos en "Bajo Riesgo" (normal_columnar, normal_intermediate, normal_superficiel) o "Alto Riesgo" (carcinoma_in_situ, light/moderate/severe_dysplastic).
   - **Modelos Integrados:** MobileNet, InceptionV3 y ResNet50 (entrenados a 20 epochs).

## Tecnologías Utilizadas

- **Backend:** Python 3, Flask.
- **Machine Learning (Tabular):** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE).
- **Visión por Computadora (Imágenes):** TensorFlow, Keras, Pillow (PIL).
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
│   ├── Herlev Dataset/                   # Carpetas de imágenes (train/test divididas por clases)
│   └── models/                           # Directorio donde se guardan los modelos pre-entrenados (.keras)
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
El servicio web requiere que los modelos `.keras` existan en la carpeta `Media/models/`. Para generarlos, debes ejecutar el script comparativo por primera vez:
```bash
python analisis_comparativo.py
```
> *Nota: Este proceso entrenará las redes neuronales durante 20 epochs y el modelo tabular. Puede demorar varios minutos dependiendo de los recursos del equipo.*

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
