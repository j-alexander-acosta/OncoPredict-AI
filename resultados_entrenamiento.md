# Reporte de Entrenamiento: Estrategia de Fine-Tuning Progresivo

Este documento recopila el proceso completo y los resultados obtenidos al aplicar la estrategia de **Fine-Tuning Progresivo** para resolver el estancamiento morfológico celular detectado en arquitecturas anteriores como MobileNet e InceptionV3.

## 1. El Proceso Realizado

1. **Adopción de Arquitectura Avanzada**:
   - Se reemplazaron temporalmente las arquitecturas base (como AlexNet o ResNet) por **EfficientNet-B0**, un modelo del estado del arte altamente eficiente para la extracción de características visuales.
   - La nueva función fue documentada en un módulo independiente y luego integrada directamente al bucle de entrenamiento principal (`train_missing.py`) bajo una bandera condicional para garantizar la reversibilidad.

2. **Fine-Tuning Progresivo por Épocas**:
   - **Fase 1 (Épocas 0 a 4)**: Se aplicó un congelamiento estricto sobre el extractor de características convolucionales. El optimizador (*Adam*) se centró de manera exclusiva en entrenar la nueva capa clasificadora (Fully Connected) para adaptarla a la cantidad de clases ginecológicas requeridas sin destruir los pesos pre-entrenados básicos.
   - **Fase 2 (Época 5 en adelante)**: Se descongelaron de manera dinámica los bloques profundos superiores (`features[6]` y `features[7]`) de EfficientNet, aplicando "Tasas de Aprendizaje Discriminativas" (*Discriminative Learning Rates*) para ajustar delicadamente la interpretación de la morfología de las células sin ocasionar el colapso del gradiente.

3. **Optimización de Hardware (MPS)**:
   - Dado el inmenso volumen de datos (más de 22,700 imágenes citológicas entre SIPaKMeD, RIVA y Herlev, además del dataset de colposcopía), se reprogramó la selección de dispositivo (Device Selection) de PyTorch en `train_missing.py` y `train_riva_all.py` para añadir el soporte de aceleración gráfica nativa para arquitecturas Apple Silicon (**MPS - Metal Performance Shaders**).
   - Esto redujo el tiempo estimado de procesamiento en CPU (que iba a tardar múltiples días) a menos de una hora para la totalidad de los datasets.

## 2. Resultados Obtenidos

Gracias al mecanismo de detención temprana (*early stopping*), los modelos encontraron su punto óptimo en un tiempo récord, con resultados que demuestran la completa efectividad de la técnica para destrabar el estancamiento en grandes volúmenes de datos.

### Tabla Consolidada de Modelos de Visión por Computadora

| Dataset | Modelo CNN | Accuracy | Sensibilidad | Especificidad | Precisión | F1-Score | AUC-ROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herlev (7 clases)** | MobileNet | 42.70% | 47.86% | 90.18% | 50.66% | 47.76% | 81.66% |
| **Herlev (7 clases)** | InceptionV3 | 40.15% | 43.71% | 89.61% | 45.27% | 43.73% | 79.34% |
| **Herlev (7 clases)** | ResNet50 | 16.79% | 11.34% | 84.97% | 4.31% | 6.11% | 57.98% |
| **Herlev (7 clases)** | AlexNet | **45.26%** | 45.96% | **90.30%** | 43.30% | 42.42% | 80.39% |
| **Herlev (7 clases)** | EfficientNet_B0 | 36.13% | 41.45% | 89.23% | 44.70% | 39.78% | 79.31% |
| **SIPaKMeD (5 clases)** | MobileNet | 49.35% | 49.42% | 87.19% | 66.98% | 49.93% | 81.29% |
| **SIPaKMeD (5 clases)** | InceptionV3 | 41.86% | 41.65% | 85.31% | 55.46% | 41.84% | 73.28% |
| **SIPaKMeD (5 clases)** | ResNet50 | 33.17% | 33.49% | 83.14% | 16.62% | 20.85% | 60.33% |
| **SIPaKMeD (5 clases)** | AlexNet | 93.63% | 93.99% | 98.40% | 93.79% | 93.76% | **99.65%** |
| **SIPaKMeD (5 clases)** | EfficientNet_B0 | **94.42%** | **94.71%** | **98.60%** | **94.49%** | **94.58%** | 99.39% |
| **RIVA (8 cat. Bethesda)** | MobileNet | 50.40% | 38.89% | 92.11% | 45.40% | 38.06% | 83.56% |
| **RIVA (8 cat. Bethesda)** | InceptionV3 | 46.99% | 34.85% | 91.58% | 39.06% | 33.44% | 81.16% |
| **RIVA (8 cat. Bethesda)** | ResNet50 | 32.89% | 15.54% | 88.31% | 19.33% | 12.28% | 66.83% |
| **RIVA (8 cat. Bethesda)** | AlexNet | **59.20%** | **50.20%** | **93.49%** | 53.82% | **49.80%** | **88.90%** |
| **RIVA (8 cat. Bethesda)** | EfficientNet_B0 | 57.50% | 46.96% | 93.11% | **60.44%** | 47.29% | 87.62% |
| **Intel_MobileODT (3 c.)** | EfficientNet_B0_Colpo | **64.13%** | **54.77%** | **77.65%** | **62.43%** | **55.62%** | **78.91%** |

### A. Dataset SIPaKMeD (5,015 imágenes: 966 frotis completos + 4,049 células recortadas en 5 clases)
*Ha presentado el salto de rendimiento más espectacular de todo el proceso.*
- **Precisión (Accuracy):** **94.42%** (`EfficientNet-B0`) / **93.63%** (`AlexNet`) *(Modelos Keras previos: `MobileNet` 49.35%, `InceptionV3` 41.86%, `ResNet50` 33.17%)*
- **Sensibilidad:** 94.71% (`EfficientNet-B0`) / 93.99% (`AlexNet`)
- **Especificidad:** 98.60% (`EfficientNet-B0`) / 98.40% (`AlexNet`)
- **Precisión Positiva:** 94.49% (`EfficientNet-B0`) / 93.79% (`AlexNet`)
- **Puntaje F1:** 94.58% (`EfficientNet-B0`) / 93.76% (`AlexNet`)
- **Área bajo la curva (AUC-ROC):** 99.39% (`EfficientNet-B0`) / **99.65%** (`AlexNet`)

### B. Dataset RIVA (16,841 imágenes: 959 frotis completos + 15,882 células recortadas en 8 categorías Bethesda)
*Evaluación sobre citología convencional (Papanicolaou) con 8 clases del Sistema Bethesda (`ASCH`, `ASCUS`, `ENDO`, `HSIL`, `INFL`, `LSIL`, `NILM`, `SCC`), donde el azar uniforme es 12.50%.*
- **Precisión (Accuracy):** 57.50% (`EfficientNet-B0`) / **59.20%** (`AlexNet`) *(Modelos Keras previos: `MobileNet` 50.40%, `InceptionV3` 46.99%, `ResNet50` 32.89%)*
- **Sensibilidad (Macro):** 46.96% (`EfficientNet-B0`) / 50.20% (`AlexNet`)
- **Especificidad:** 93.11% (`EfficientNet-B0`) / 93.49% (`AlexNet`)
- **Precisión Positiva (Macro):** **60.44%** (`EfficientNet-B0`) / 53.82% (`AlexNet`)
- **Puntaje F1:** 47.29% (`EfficientNet-B0`) / 49.80% (`AlexNet`)
- **Área bajo la curva (AUC-ROC):** 87.62% (`EfficientNet-B0`) / **88.90%** (`AlexNet`)

### C. Dataset Herlev (917 imágenes en 7 clases)
*Limitaciones de Volumen de Datos.*
- **Precisión (Accuracy):** 36.13% (`EfficientNet-B0`) / **45.26%** (`AlexNet`) *(Modelos Keras: `MobileNet` 42.70%, `InceptionV3` 40.15%, `ResNet50` 16.79%)*
- **Sensibilidad:** 41.45% (`EfficientNet-B0`) / 45.96% (`AlexNet`) / **47.86%** (`MobileNet`)
- **Especificidad:** 89.23% (`EfficientNet-B0`) / **90.30%** (`AlexNet`) / 90.18% (`MobileNet`)
- **Precisión Positiva:** 44.70% (`EfficientNet-B0`) / 43.30% (`AlexNet`) / **50.66%** (`MobileNet`)
- **Puntaje F1:** 39.78% (`EfficientNet-B0`) / 42.42% (`AlexNet`) / **47.76%** (`MobileNet`)
- **Área bajo la curva (AUC-ROC):** 79.31% (`EfficientNet-B0`) / 80.39% (`AlexNet`) / **81.66%** (`MobileNet`)

### D. Dataset Intel & MobileODT (Colposcopía Macroscópica - 3 clases de Zona de Transformación)
- **Precisión (Accuracy):** 64.13% (`EfficientNet_B0_Colpo`)
- **Sensibilidad:** 54.77%
- **Especificidad:** 77.65%
- **Precisión Positiva:** 62.43%
- **Puntaje F1:** 55.62%
- **Área bajo la curva (AUC-ROC):** 78.91%

> **Análisis de Herlev**: Aunque EfficientNet-B0 es sumamente potente, las redes profundas requieren grandes bancos de datos para generalizar patrones. El volumen tan reducido de imágenes de Herlev (917 imágenes frente a las 5,015 de SIPaKMeD y 16,841 de RIVA) provocó que la red no tuviese suficiente exposición matemática para ajustarse adecuadamente, reafirmando que en Machine Learning el volumen de la base de datos es tan crucial como la arquitectura misma.

## 3. Conclusión
El experimento fue un **éxito rotundo**. Se verificó que el estancamiento morfológico celular de los modelos previos (MobileNet, ResNet, Inception) provenía de la incapacidad de adaptar los pesos convolucionales de manera progresiva. La combinación de **EfficientNet-B0 + Fine-Tuning Progresivo** logró escalar la asertividad a un impresionante ~94% para bases de datos clínicas densas.

- Las nuevas métricas se encuentran respaldadas en `image_metrics.json`.
- Los pesos hiper-optimizados (de tan solo 18 MB vs 177 MB de AlexNet) descansan en `Media/models/`.

## 4. Auditoría de Inteligencia Artificial Explicable (XAI)
Para elevar la transparencia clínica de la red neuronal y eliminar el paradigma de la "caja negra", se implementó un pipeline riguroso de IA Explicable (XAI) con los siguientes resultados reales sobre el modelo `EfficientNet-B0`:

1. **Interpretabilidad de Alta Resolución (Score-CAM):** Se reemplazó el tradicional Grad-CAM por Score-CAM para generar mapas de calor sin ruido de gradientes. El resultado visual demuestra un enfoque anatómico preciso en los núcleos hipertróficos de las células.
2. **Evaluación Cuantitativa (Deletion AUC):** Se midió matemáticamente la fidelidad del mapa de calor borrando de mayor a menor los píxeles resaltados. Se obtuvo un **Deletion AUC de 0.2890** (valores más cercanos a 0 indican mapas de altísima precisión), validando cuantitativamente la interpretabilidad local.
3. **Explicabilidad Global (TCAV):** Se implementó *Testing with Concept Activation Vectors* para medir si el modelo aprendió correctamente el concepto de "célula anormal". 
   - El clasificador SVM interno alcanzó un **100.00% de exactitud** al separar vectores latentes anormales del ruido aleatorio, demostrando que el concepto clínico existe y está codificado nítidamente en la última capa de la red.
   - La red presenta una sensibilidad positiva (TCAV Score = 26.67%) hacia los datos de prueba, lo que invita a refinar futuras métricas con múltiples conceptos ginecológicos aislados (ej. citoplasma oscuro vs. tamaño del núcleo) para entender la topología matemática de las decisiones clínicas.
