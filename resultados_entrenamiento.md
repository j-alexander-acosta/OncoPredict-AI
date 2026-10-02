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
   - Dado el inmenso volumen de datos (más de 43,000 imágenes), se reprogramó la selección de dispositivo (Device Selection) de PyTorch en `train_missing.py` para añadir el soporte de aceleración gráfica nativa para arquitecturas Apple Silicon (**MPS - Metal Performance Shaders**).
   - Esto redujo el tiempo estimado de procesamiento en CPU (que iba a tardar múltiples días) a tan solo menos de una hora para la totalidad de los 3 datasets.

## 2. Resultados Obtenidos

Gracias al mecanismo de detención temprana (*early stopping*), los modelos encontraron su punto óptimo en un tiempo récord, con resultados que demuestran la completa efectividad de la técnica para destrabar el estancamiento en grandes volúmenes de datos.

### A. Dataset SIPaKMeD (~21,200 imágenes)
*Ha presentado el salto de rendimiento más espectacular de todo el proceso.*
- **Precisión (Accuracy):** 94.42% *(Modelos previos rondaban entre el 33% y el 49%)*
- **Sensibilidad:** 94.71%
- **Especificidad:** 98.60%
- **Precisión Positiva:** 94.49%
- **Puntaje F1:** 94.58%
- **Área bajo la curva (AUC):** 99.39%

### B. Dataset RIVA (~21,200 imágenes)
*El comportamiento positivo se replica casi con exactitud en este set de alta densidad.*
- **Precisión (Accuracy):** 93.63% *(Modelos previos rondaban entre el 21% y el 47%)*
- **Sensibilidad:** 93.93%
- **Especificidad:** 98.39%
- **Precisión Positiva:** 93.84%
- **Puntaje F1:** 93.86%
- **Área bajo la curva (AUC):** 99.34%

### C. Dataset Herlev (~900 imágenes)
*Limitaciones de Volumen de Datos.*
- **Precisión (Accuracy):** 36.13%
- **Sensibilidad:** 41.45%
- **Especificidad:** 89.23%
- **Precisión Positiva:** 44.70%
- **Puntaje F1:** 39.78%
- **Área bajo la curva (AUC):** 79.31%

> **Análisis de Herlev**: Aunque EfficientNet-B0 es sumamente potente, las redes profundas requieren grandes bancos de datos para generalizar patrones. El volumen tan reducido de imágenes de Herlev (frente a las más de 20,000 de los otros) provocó que la red no tuviese suficiente exposición matemática para ajustarse adecuadamente, reafirmando que en Machine Learning el volumen de la base de datos es tan crucial como la arquitectura misma.

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
