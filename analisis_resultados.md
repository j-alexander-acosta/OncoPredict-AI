# Análisis de Métricas de Evaluación (Conjunto de Prueba)

A continuación presento el análisis interpretativo del rendimiento de los modelos en base a los resultados consolidados de inferencia clínica (Tabular) y análisis de imagen (CV) usando los datasets **UCI**, **Herlev**, **SIPaKMeD** y **RIVA**.

## 1. Análisis de Modelos Tabulares (Dataset UCI)

**Paradoja del Accuracy y Peligro Clínico:** Los modelos exhiben un "Accuracy" engañosamente alto (hasta 93.80%), pero fallan críticamente en lo más importante: detectar la enfermedad.

*   **Precisión Engañosa:** Modelos como **XGBoost** y **Random Forest** alcanzan una exactitud global del 93.80%. Sin embargo, al observar la **Sensibilidad (Recall)** vemos que es apenas del 12.50%. Esto significa que de cada 100 mujeres que **sí tienen** riesgo de cáncer, el modelo solo está detectando a 12, dejando ir a 88 (Falsos Negativos).
*   **Falla del SVM:** El algoritmo **SVM** tiene un Accuracy del 89.15%, pero su Sensibilidad, Precisión y F1-Score son del **0.00%**. Esto indica que el modelo simplemente predice "Sano" para todas las pacientes. La extrema Especificidad (95.04%) es síntoma de que el modelo ignoró la clase minoritaria por completo.
*   **Conclusión de Fase 1:** A pesar de haber implementado SMOTE (balanceo), los modelos están altamente sesgados hacia la clase "Sano". Para entornos clínicos, es preferible el **XGBoost** por tener el mejor AUC-ROC (76.76%), pero requiere urgentemente **ajustar el umbral de decisión (threshold)** para tolerar más falsos positivos a cambio de atrapar todos los verdaderos positivos.

---

## 2. Análisis de Visión por Computadora (Transfer Learning)

Los resultados revelan patrones consistentes a través de los tres datasets de citologías (Herlev, SIPaKMeD, RIVA).

### Rendimiento por Arquitectura de Red
*   🏆 **AlexNet (El nuevo campeón en SIPaKMeD y RIVA):** Tras su reciente entrenamiento en los datasets de **SIPaKMeD** y **RIVA**, la red AlexNet demostró resultados sobresalientes, superando a todos los demás modelos. Alcanzó una Exactitud del **90.84%** y un impresionante AUC-ROC del **99.21%** en SIPaKMeD, y un 89.51% de Exactitud con 98.61% de AUC-ROC en RIVA. Estos resultados demuestran la eficacia de PyTorch y el fine-tuning para tareas de clasificación morfológica, logrando un balance casi perfecto entre sensibilidad y especificidad. Sin embargo, en **Herlev** el rendimiento fue notablemente inferior (32% Accuracy), denotando que no logró generalizar sobre la menor cantidad y/o contraste de esas imágenes.
*   🥈 **MobileNet (Sólido y consistente):** Fue consistentemente el segundo mejor modelo a través de todos los datasets y el **mejor en Herlev**. Su pico de rendimiento fue en RIVA (Accuracy: 54.35%, F1: 54.55%). Además, en Herlev mostró una asombrosa capacidad de discriminación latente con un **AUC-ROC del 81.42%**. 
*   🥉 **InceptionV3:** Mantuvo resultados decentes y competitivos, con métricas de Accuracy rondando el 42% - 47%.
*   📉 **ResNet50:** Su rendimiento colapsó en los tres datasets (Accuracy inferior al 22% y Sensibilidades paupérrimas del ~15%). Esto sugiere fuertemente que **20 épocas no fueron suficientes** para que esta red (que es muy profunda) ajustara sus densas capas finales, o que congelar todas sus capas base impidió que extrajera características del dominio médico.

### Diferencias entre Datasets
*   **RIVA** demostró ser el dataset con el cual las redes obtuvieron métricas más altas de Accuracy (MobileNet: 54.35%), seguido de **SIPaKMeD** (50.85%) y finalmente **Herlev** (43.80%). Esto podría deberse a variaciones en la resolución de las imágenes, el contraste de tintura de las células, o el equilibrio de clases dentro de las carpetas de RIVA y SIPaKMeD comparado con Herlev.

**Alta Especificidad vs Baja Sensibilidad en CV:** Al igual que en tabular, las CNN (MobileNet e InceptionV3) alcanzan Especificidades del ~85-90% pero Sensibilidades que rondan solo el ~50%. Tienen excelente capacidad para descartar tejido sano, pero dudan al clasificar tejido enfermo. 

---

## 3. Conclusiones y Recomendaciones Claves

1.  **Ajuste del Umbral Clínico (Threshold):** La métrica de AUC-ROC (~81% en MobileNet y ~76% en XGBoost) demuestra que los modelos **sí han aprendido a separar** a las pacientes sanas de las enfermas matemáticamente. El bajo Recall (Sensibilidad) se soluciona moviendo el umbral de predicción de la probabilidad. Actualmente, si el modelo dice "51% de cáncer", lo clasifica como enfermo; en oncología, deberíamos clasificar como "Alto Riesgo" incluso a pacientes que el modelo arroje "20% de probabilidad".
2.  **Fine-Tuning Profundo (Imágenes):** En lugar de solo entrenar las últimas capas, recomiendo descongelar el 30% superior de capas de MobileNet e InceptionV3 y re-entrenar con una tasa de aprendizaje (Learning Rate) muy baja (`1e-5`). 
3.  **Descartar ResNet50 en este estado:** Hasta no dotarlo de más épocas de entrenamiento (ej. 100 epochs), ResNet50 no es confiable para este caso de uso médico específico bajo *Transfer Learning* de características genéricas.
