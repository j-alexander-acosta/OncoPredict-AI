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

### Rendimiento por Arquitectura de Red (Tras re-entrenamiento masivo con Early Stopping)
*   🏆 **AlexNet (El indiscutible campeón en SIPaKMeD y RIVA):** Tras el re-entrenamiento intensivo con paciencia, la red AlexNet rompió sus propios récords. Alcanzó una Exactitud del **93.63%** y un impresionante AUC-ROC del **99.65%** en SIPaKMeD, y un 92.83% de Exactitud con 99.47% de AUC-ROC en RIVA. Estos resultados demuestran la superioridad de la arquitectura de PyTorch implementada. Además, logró mejorar su rendimiento en **Herlev**, subiendo del 32% al 45.26%.
*   🥈 **MobileNet (Estabilidad en Keras):** Fue consistentemente el mejor modelo de TensorFlow/Keras a través de todos los datasets. Logró rozar el 50% de Exactitud en SIPaKMeD y mantiene AUC-ROCs superiores al 81%, denotando buena discriminación latente. 
*   🥉 **InceptionV3:** Mantuvo resultados moderados, con métricas de Exactitud estancadas alrededor del 40% - 42% en todos los frentes.
*   📉 **ResNet50 (Deficiencia Arquitectónica):** A pesar de darle un máximo de 50 epochs, el mecanismo de Early Stopping detuvo su entrenamiento prematuramente (ej. en la época 6) debido a que su pérdida de validación dejó de mejorar muy rápido. Su rendimiento mejoró ligeramente (subió a 33% en SIPaKMeD), pero sigue colapsado. Esto sugiere fuertemente que **no era un problema de cantidad de epochs**, sino que congelar todas sus capas base impide que extraiga características del dominio médico, o su capa densa superior es muy plana.

### Diferencias entre Datasets
*   **RIVA** demostró ser el dataset con el cual las redes obtuvieron métricas más altas de Accuracy (MobileNet: 54.35%), seguido de **SIPaKMeD** (50.85%) y finalmente **Herlev** (43.80%). Esto podría deberse a variaciones en la resolución de las imágenes, el contraste de tintura de las células, o el equilibrio de clases dentro de las carpetas de RIVA y SIPaKMeD comparado con Herlev.

**Alta Especificidad vs Baja Sensibilidad en CV:** Al igual que en tabular, las CNN (MobileNet e InceptionV3) alcanzan Especificidades del ~85-90% pero Sensibilidades que rondan solo el ~50%. Tienen excelente capacidad para descartar tejido sano, pero dudan al clasificar tejido enfermo. 

---

## 3. Conclusiones y Recomendaciones Claves

1.  **Ajuste del Umbral Clínico (Threshold):** La métrica de AUC-ROC (~81% en MobileNet y ~76% en XGBoost) demuestra que los modelos **sí han aprendido a separar** a las pacientes sanas de las enfermas matemáticamente. El bajo Recall (Sensibilidad) se soluciona moviendo el umbral de predicción de la probabilidad. Actualmente, si el modelo dice "51% de cáncer", lo clasifica como enfermo; en oncología, deberíamos clasificar como "Alto Riesgo" incluso a pacientes que el modelo arroje "20% de probabilidad".
2.  **Fine-Tuning Profundo (Imágenes):** En lugar de solo entrenar las últimas capas, recomiendo descongelar el 30% superior de capas de MobileNet e InceptionV3 y re-entrenar con una tasa de aprendizaje (Learning Rate) muy baja (`1e-5`). 
3.  **El problema real de ResNet50:** Ya confirmamos vía Early Stopping que darle más epochs a ResNet50 no sirve de nada en su configuración actual. Para hacerlo competitivo, se debe descongelar parcial o totalmente su bloque convolucional final.
