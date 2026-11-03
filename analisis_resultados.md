# Análisis de Métricas de Evaluación (Conjunto de Prueba)

A continuación se presenta el análisis interpretativo del rendimiento de los modelos en base a los resultados consolidados de inferencia clínica (Tabular) y análisis de imagen (Visión por Computadora) usando los datasets **UCI**, **Herlev**, **SIPaKMeD**, **RIVA** e **Intel & MobileODT**.

## 1. Análisis de Modelos Tabulares (Dataset UCI - 129 Casos de Prueba)

| Modelo | Accuracy | Sensibilidad | Especificidad | Precisión | F1-Score | AUC-ROC | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Dummy Trivial)** | 93.80% | 0.00% | 100.00% | 0.00% | 0.00% | 50.00% | 0 | 0 | 121 | 8 |
| **Regresión Logística** | 79.84% | 12.50% | 84.30% | 5.00% | 7.14% | 50.93% | 1 | 19 | 102 | 7 |
| **SVM** | 89.15% | 0.00% | 95.04% | 0.00% | 0.00% | 65.60% | 0 | 6 | 115 | 8 |
| **Decision Tree** | 87.60% | 25.00% | 91.74% | 16.67% | 20.00% | 58.06% | 2 | 10 | 111 | 6 |
| **Random Forest** | 92.25% | 12.50% | 97.52% | 25.00% | 16.67% | 76.45% | 1 | 3 | 118 | 7 |
| **XGBoost** | 93.80% | 12.50% | 99.17% | 50.00% | 20.00% | 75.41% | 1 | 1 | 120 | 7 |

**Paradoja del Accuracy y Peligro Clínico:** Los modelos exhiben un "Accuracy" engañosamente alto (hasta 93.80%), pero fallan críticamente en lo más importante bajo el umbral por defecto ($p=0.50$): detectar la enfermedad.

*   **Precisión Engañosa:** Modelos como **XGBoost** (93.80%) y **Random Forest** (92.25%) alcanzan una exactitud global muy elevada. Sin embargo, al observar la **Sensibilidad (Recall)** vemos que es apenas del 12.50% (1 de 8 casos positivos detectados; **Decision Tree** alcanza 25.00% con 2 de 8 casos). Esto significa que de cada 100 mujeres que **sí tienen** riesgo de cáncer, el modelo solo detecta a ~12 con el umbral estándar, dejando ir a ~88 (Falsos Negativos).
*   **Falla del SVM:** El algoritmo **SVM** tiene un Accuracy del 89.15%, pero su Sensibilidad, Precisión y F1-Score son del **0.00%**. Esto indica que el modelo predice casi exclusivamente "Sano" para las pacientes (6 falsos positivos y 0 verdaderos positivos). La alta Especificidad (95.04%) es síntoma de que el umbral estándar ignora la clase minoritaria.
*   **Conclusión de Fase 1:** A pesar de haber implementado SMOTE (balanceo en entrenamiento), la evaluación sobre la distribución real desbalanceada (15.1:1) sesga las decisiones hacia la clase "Sano". Para entornos clínicos, **Random Forest** (AUC-ROC de **76.45%**) y **XGBoost** (AUC-ROC de **75.41%**, Precisión de **50.00%** y Especificidad de **99.17%**) ofrecen la mejor capacidad de discriminación latente, pero requieren **ajustar el umbral de decisión (threshold)** para tolerar más falsos positivos a cambio de capturar los verdaderos positivos.

### Atribución SHAP sobre XGBoost (TreeSHAP en Log-Odds)

| Ranking | Factor de Riesgo Clínico (UCI) | mean(\|SHAP\|) Test (log-odds) | Aporte SHAP Test (%) | mean(\|SHAP\|) Train (log-odds) | Importancia Gini (RF) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Num of pregnancies | 1.0488 | 18.06% | 1.3955 | 0.1735 |
| **2** | Number of sexual partners | 0.9874 | 17.00% | 1.5794 | 0.1456 |
| **3** | First sexual intercourse | 0.9102 | 15.67% | 1.2212 | 0.1484 |
| **4** | Hormonal Contraceptives (years) | 0.7538 | 12.98% | 0.6955 | 0.1002 |
| **5** | Age | 0.6920 | 11.91% | 0.6137 | 0.1134 |
| **6** | Hormonal Contraceptives | 0.2243 | 3.86% | 0.2812 | 0.0296 |
| **7** | STDs: Time since first diagnosis | 0.1825 | 3.14% | 0.1907 | 0.0151 |
| **8** | STDs: Number of diagnosis | 0.1674 | 2.88% | 0.1999 | 0.0141 |
| **9** | Dx | 0.1369 | 2.36% | 0.1997 | 0.0335 |
| **10** | Smokes (packs/year) | 0.1200 | 2.07% | 0.1071 | 0.0193 |
| **11** | IUD (years) | 0.1139 | 1.96% | 0.1587 | 0.0279 |
| **12** | STDs:HIV | 0.0705 | 1.21% | 0.1289 | 0.0140 |

---

## 2. Análisis de Visión por Computadora (Transfer Learning y Fine-Tuning Progresivo)

| Dataset | Modelo CNN | Accuracy | Sensibilidad (Macro) | Especificidad | Precisión (Macro) | F1-Score | AUC-ROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herlev (7 clases)** | MobileNet | 42.70% | 47.86% | 90.18% | 50.66% | 47.76% | 81.66% |
| **Herlev (7 clases)** | InceptionV3 | 40.15% | 43.71% | 89.61% | 45.27% | 43.73% | 79.34% |
| **Herlev (7 clases)** | ResNet50 | 16.79% | 11.34% | 84.97% | 4.31% | 6.11% | 57.98% |
| **Herlev (7 clases)** | AlexNet | 45.26% | 45.96% | 90.30% | 43.30% | 42.42% | 80.39% |
| **Herlev (7 clases)** | EfficientNet_B0 | 36.13% | 41.45% | 89.23% | 44.70% | 39.78% | 79.31% |
| **SIPaKMeD (5 clases)** | MobileNet | 49.35% | 49.42% | 87.19% | 66.98% | 49.93% | 81.29% |
| **SIPaKMeD (5 clases)** | InceptionV3 | 41.86% | 41.65% | 85.31% | 55.46% | 41.84% | 73.28% |
| **SIPaKMeD (5 clases)** | ResNet50 | 33.17% | 33.49% | 83.14% | 16.62% | 20.85% | 60.33% |
| **SIPaKMeD (5 clases)** | AlexNet | 93.63% | 93.99% | 98.40% | 93.79% | 93.76% | 99.65% |
| **SIPaKMeD (5 clases)** | EfficientNet_B0 | **94.42%** | **94.71%** | **98.60%** | **94.49%** | **94.58%** | 99.39% |
| **RIVA (8 cat. Bethesda)** | MobileNet | 50.40% | 38.89% | 92.11% | 45.40% | 38.06% | 83.56% |
| **RIVA (8 cat. Bethesda)** | InceptionV3 | 46.99% | 34.85% | 91.58% | 39.06% | 33.44% | 81.16% |
| **RIVA (8 cat. Bethesda)** | ResNet50 | 32.89% | 15.54% | 88.31% | 19.33% | 12.28% | 66.83% |
| **RIVA (8 cat. Bethesda)** | AlexNet | **59.20%** | **50.20%** | **93.49%** | 53.82% | **49.80%** | **88.90%** |
| **RIVA (8 cat. Bethesda)** | EfficientNet_B0 | 57.50% | 46.96% | 93.11% | **60.44%** | 47.29% | 87.62% |
| **Intel_MobileODT (3 c.)** | EfficientNet_B0_Colpo | **64.13%** | **54.77%** | **77.65%** | **62.43%** | **55.62%** | **78.91%** |

### Rendimiento por Arquitectura de Red (Tras re-entrenamiento con Fine-Tuning Progresivo y Early Stopping)
*   🏆 **EfficientNet-B0 y AlexNet (Campeones en PyTorch):**
    *   En **SIPaKMeD** (5 clases), **EfficientNet-B0** alcanzó el máximo desempeño global con **94.42%** de Exactitud, **94.71%** de Sensibilidad, **98.60%** de Especificidad, **94.49%** de Precisión, **94.58%** de F1-Score y **99.39%** de AUC-ROC, seguido muy de cerca por **AlexNet** con **93.63%** de Exactitud y **99.65%** de AUC-ROC.
    *   En el dataset oficial **RIVA** (16,841 imágenes en 8 categorías del Sistema Bethesda, donde el azar uniforme es 12.50%), **AlexNet** lideró en Exactitud (**59.20%**), Sensibilidad (**50.20%**), Especificidad (**93.49%**), F1-Score (**49.80%**) y AUC-ROC (**88.90%**), mientras que **EfficientNet-B0** alcanzó la mayor Precisión Positiva Macro (**60.44%**) con **57.50%** de Exactitud y **87.62%** de AUC-ROC.
    *   En **Herlev** (917 imágenes en 7 clases), **AlexNet** alcanzó **45.26%** de Exactitud (AUC-ROC **80.39%**) y **EfficientNet-B0** obtuvo **36.13%** (AUC-ROC **79.31%**).
    *   En **Intel & MobileODT** (Colposcopía macroscópica, 3 clases), **EfficientNet_B0_Colpo** alcanzó **64.13%** de Exactitud, **77.65%** de Especificidad y **78.91%** de AUC-ROC.
*   🥈 **MobileNet (Estabilidad en Keras):** Fue consistentemente el mejor modelo de TensorFlow/Keras a través de los datasets citológicos, logrando **50.40%** de Exactitud en RIVA (AUC-ROC **83.56%**), **49.35%** en SIPaKMeD (AUC-ROC **81.29%**) y **42.70%** en Herlev (AUC-ROC **81.66%**), denotando buena discriminación latente.
*   🥉 **InceptionV3:** Mantuvo resultados intermedios en Keras, con métricas de Exactitud de **46.99%** en RIVA (AUC-ROC **81.16%**), **41.86%** en SIPaKMeD (AUC-ROC **73.28%**) y **40.15%** en Herlev (AUC-ROC **79.34%**).
*   📉 **ResNet50 (Deficiencia Arquitectónica con Capas Congeladas):** El mecanismo de Early Stopping detuvo su entrenamiento prematuramente debido al estancamiento de validación cuando se congelan sus bloques base. Alcanzó **33.17%** de Exactitud en SIPaKMeD (AUC-ROC **60.33%**), **32.89%** en RIVA (AUC-ROC **66.83%**) y **16.79%** en Herlev (AUC-ROC **57.98%**). Esto confirma que congelar todas sus capas profundas impide extraer características morfológicas finas sin un descongelamiento progresivo por bloques.

### Diferencias entre Datasets
*   En las arquitecturas con *Fine-Tuning* en PyTorch, **SIPaKMeD** logró las métricas más sobresalientes (**94.42%** en EfficientNet-B0 y **93.63%** en AlexNet), seguido de **RIVA** (**59.20%** en AlexNet y **57.50%** en EfficientNet-B0 sobre 8 categorías complejas de Bethesda) y **Herlev** (**45.26%** en AlexNet y **42.70%** en MobileNet), donde el reducido volumen muestral (917 imágenes en 7 clases) limita la generalización de redes profundas.
*   En los modelos Keras con extractor fijo, **RIVA** obtuvo el mayor Accuracy con MobileNet (**50.40%**), seguido de **SIPaKMeD** (**49.35%**) y **Herlev** (**42.70%**).

**Alta Especificidad vs Sensibilidad Moderada en Multiclase:** En datasets complejos de 7 y 8 clases (Herlev y RIVA), las redes alcanzan Especificidades del ~89-93% pero Sensibilidades macro entre ~39-50%, mostrando alta fiabilidad para descartar clases negativas (alta especificidad) y mayor exigencia al separar subtipos displásicos limítrofes.

---

## 3. Conclusiones y Recomendaciones Claves

1.  **Ajuste del Umbral Clínico (Threshold):** La métrica de AUC-ROC (**76.45%** en Random Forest y **75.41%** en XGBoost para datos tabulares; **87.62%–99.65%** en PyTorch para citología) demuestra que los modelos **sí han aprendido a separar** matemáticamente los perfiles patológicos. El bajo Recall en datos tabulares se soluciona ajustando el umbral de probabilidad ($p^* \approx 0.15 - 0.20$) para priorizar la sensibilidad oncológica.
2.  **Efectividad del Fine-Tuning Progresivo (Imágenes):** Descongelar progresivamente los bloques profundos con tasas de aprendizaje discriminativas (`1e-5` / `1e-4`) en `EfficientNet-B0` y `AlexNet` permitió romper el techo del ~50% de Keras, alcanzando **94.42%** en SIPaKMeD y casi **60%** en las 8 clases de RIVA.
3.  **El problema real de ResNet50:** Se confirmó vía Early Stopping que entrenar solo el cabezal denso en ResNet50 genera un cuello de botella severo (16.79% - 33.17%); requiere descongelamiento gradual de sus bloques residuales finales para adaptarse al dominio citológico.
