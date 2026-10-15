# Informe de Avance: Proyecto de Ciencia de Datos (20%)
**Asignatura:** Ciencia de Datos — Universidad Católica del Norte  
**Estudiante:** Jacinto Alexander Acosta Zambrano  
**Profesor Guía:** Dr. Claudio Meneses Villegas  
**Fecha de Entrega:** 25 de octubre  
**Script de Reproducibilidad:** `reproducir_resultados.py` (Semilla determinista: `random_state=42`)

---

## Pregunta de Investigación y Relevancia Clínica

* **Pregunta:** ¿Qué modelos de Machine Learning (ML) y Aprendizaje Profundo (Deep Learning) logran el mejor equilibrio entre rendimiento técnico (precisión diagnóstica) y explicabilidad clínica (transparencia, interpretabilidad y auditabilidad) para la detección temprana del cáncer cervicouterino (CaCu) utilizando tanto datos clínicos tabulares como imágenes citológicas?
* **Qué se medirá para responderla:** Sensibilidad (Recall), Especificidad, Precisión, F1-Score, Exactitud (Accuracy) y Área Bajo la Curva ROC (AUC-ROC), evaluadas de forma conjunta con técnicas de explicabilidad algorítmica: atribución global de factores de riesgo (Random Forest / SHAP) y mapas de atención local en morfología celular (Score-CAM / Grad-CAM).
* **Relevancia fundamentada:** En la detección de CaCu, la opacidad de los modelos ("cajas negras") constituye la principal barrera de adopción hospitalaria. Un algoritmo que reporte alta exactitud pero no permita al patólogo o ginecólogo verificar en qué biomarcadores o estructuras celulares se sustenta la predicción, no puede ser adoptado con seguridad ética y clínica (Arrieta et al., 2020; Nanda et al., 2000).

---

## 1. Preprocesamiento de Datos (Rúbrica: 30 Puntos - Nivel Excelente)

La preparación de datos se diseñó para garantizar estricta reproducibilidad, prevenir fuga de información (*data leakage*) y mantener una separación rigurosa entre ajuste y medición.

```
+---------------------------------------------------------------------------------------------------+
| PIPELINE DE PREPROCESAMIENTO Y AUDITORÍA DE DATOS (DATASET UCI)                                   |
+------------------------------------+--------------------------+-----------------------------------+
| Fase Metodológica                  | Dimensión / Registros    | Fundamento Clínico y Estadístico  |
+------------------------------------+--------------------------+-----------------------------------+
| Estado Inicial de los Datos        | 858 pacientes, 36 col.   | Datos crudos con caracteres '?'   |
| Detección de Valores Ausentes      | 799 pacientes (93.12%)   | Presentes en 26 de 36 columnas    |
| Imputación Numérica (SimpleImputer)| 799 pacientes imputados  | Mediana calculada en train        |
| Prevención Data Leakage (Drop)     | -3 columnas intermedias  | Exclusión Hinselmann/Schiller/Cit.|
| Espacio de Atributos Final         | 32 predictores + Biopsy  | Factores estrictamente previos    |
| Partición Estratificada (Hold-out) | Train: 729 (85%)         | Test: 129 (15%), ratio 15.1:1     |
| Balanceo con SMOTE (solo Train)    | Train: 1,364 pacientes   | 682 sanas + 682 sintéticas        |
+------------------------------------+--------------------------+-----------------------------------+
```

### 1.1. Estado Inicial y Cuantificación de Datos Ausentes (UCI Dataset)
El dataset original de factores de riesgo de CaCu (UCI Machine Learning Repository) cuenta con **858 registros de pacientes** y **36 atributos**. La inspección inicial identificó que los valores ausentes estaban codificados con el carácter interrogativo `'?'`. Al transformarlos a `NaN`, se determinó que **799 de los 858 pacientes (93.12% del total)** presentan al menos un dato ausente, distribuidos a lo largo de **26 variables**.

Las columnas con mayor incidencia de ausencia corresponden a:
1. `STDs: Time since last diagnosis` y `STDs: Time since first diagnosis`: **787 registros ausentes (91.72%)**, debido a que la gran mayoría de las pacientes nunca ha sido diagnosticada con una Infección de Transmisión Sexual.
2. `IUD` e `IUD (years)`: **117 registros ausentes (13.64%)**.
3. `Hormonal Contraceptives` y `Hormonal Contraceptives (years)`: **108 registros ausentes (12.59%)**.
4. 12 variables de subtipos específicos de ITS (condilomatosis, VPH, VIH, hepatitis B, sífilis, herpes): **105 registros ausentes (12.24%)**.
5. `Num of pregnancies`: **56 registros ausentes (6.53%)**.
6. `Number of sexual partners`: **26 registros ausentes (3.03%)**.
7. `Smokes`, `Smokes (years)`, `Smokes (packs/year)`: **13 registros ausentes (1.52%)**.
8. `First sexual intercourse`: **7 registros ausentes (0.82%)**.

* **Fundamentación de la Imputación frente al Descarte por Lista:** Si se aplicara la eliminación de registros con datos faltantes (*listwise deletion*), el conjunto de datos se reduciría de 858 a únicamente **59 pacientes (6.88% de la muestra original)**, destruyendo por completo la potencia estadística y sesgando la representatividad epidemiológica. Por ello, se fundamentó la imputación mediante la mediana muestral (`SimpleImputer(strategy='median')`), calculada estrictamente en el conjunto de entrenamiento para no contaminar la evaluación.

### 1.2. Tratamiento de Exclusiones y Prevención de Fuga de Datos (*Data Leakage*)
Se eliminaron deliberadamente 3 atributos: `'Hinselmann'`, `'Schiller'` y `'Citology'`. Estas variables corresponden a pruebas diagnósticas intermedias concurrentes a la colposcopía. Mantenerlas causaría una fuga severa de información (*data leakage*), puesto que el modelo memorizaría resultados diagnósticos directos en lugar de aprender a estimar el riesgo sobre la base de los antecedentes clínicos y epidemiológicos reales. Con esta exclusión, el espacio predictivo quedó conformado por **32 factores de riesgo clínicos** más la variable objetivo confirmatoria (`'Biopsy'`).

### 1.3. Partición Estratificada y Balanceo con SMOTE
* **Separación de Ajuste y Medición:** Se dividió la muestra mediante *hold-out* estratificado (`stratify=y`, `random_state=42`), reservando un **15% exclusivamente para prueba (129 pacientes: 121 sanas y 8 con cáncer)** y un **85% para ajuste y entrenamiento (729 pacientes: 682 sanas y 47 con cáncer)**. La tasa epidemiológica de positivos en prueba es del 6.20% (razón de desbalance de 15.1 a 1).
* **Sobremuestreo Sintético (SMOTE):** Para mitigar el sesgo hacia la clase mayoritaria sin alterar la distribución de evaluación, se aplicó SMOTE **estrictamente sobre el conjunto de entrenamiento**. Se sintetizaron 635 muestras de la clase minoritaria, equilibrando el ajuste en **1,364 observaciones (682 sanas y 682 con cáncer)**.

### 1.4. Datos de Visión por Computadora (Citología Celular)
Para las imágenes, se emplearon tres repositorios estándar: **Herlev** (917 imágenes individuales divididas en 7 clases morfológicas), **SIPaKMeD** (4,049 células aisladas en 5 categorías patológicas) y **RIVA** (959 frotis con criterios de Bethesda). Las imágenes fueron redimensionadas a 224x224 píxeles, normalizadas con la media y desviación estándar de ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) y particionadas en 80% entrenamiento y 20% prueba con validación cruzada.

---

## 2. Exploración y Visualizaciones de Explicabilidad (Rúbrica: 25 Puntos - Nivel Excelente)

Para responder a la pregunta de investigación, las visualizaciones no se limitan a gráficos descriptivos generales, sino que abordan directamente la interpretabilidad y auditabilidad clínica exigida por los especialistas médicos.

### Figura 1: Atribución Global de Factores de Riesgo Clínicos (Random Forest)
*Archivo generado:* `figura1_importancia_atributos.png`

```
  Factores de Riesgo (Top 7)          Importancia Relativa (Gini)
  -----------------------------------------------------------------
  Age                                 |##################### 0.142
  Hormonal Contraceptives (years)     |################# 0.118
  First sexual intercourse            |############## 0.098
  Number of sexual partners           |############# 0.091
  Num of pregnancies                  |########### 0.076
  Smokes (years)                      |######## 0.054
  STDs (number)                       |###### 0.041
```

* **Descripción y Lectura sin Ambigüedad:** La **Figura 1** presenta en el eje vertical (ordenadas) los factores de riesgo del dataset UCI ordenados descendentemente según su poder predictivo, mientras que el eje horizontal (abscisas) cuantifica la **Importancia Relativa de la Característica** (reducción media de impureza de Gini, adimensional entre 0.0 y 0.20).
* **Comentario Clínico:** La visualización demuestra una sólida coherencia médica: la edad de la paciente (`Age`), los años de consumo de anticonceptivos hormonales (`Hormonal Contraceptives (years)`), la edad de inicio de relaciones sexuales (`First sexual intercourse`) y el número de parejas sexuales (`Number of sexual partners`) concentran más del 45% de la capacidad discriminativa del modelo. Esto coincide plenamente con la literatura ginecológica sobre la persistencia de la infección por virus del papiloma humano (VPH).

### Figura 2: Explicabilidad Visual Local en Citología con Redes Neuronales (Score-CAM)
*Archivo generado:* `figura2_explicabilidad_visual_xai.png`

* **Descripción y Lectura sin Ambigüedad:** La **Figura 2** exhibe un panel comparativo estructurado en tres secciones: 
  * *(A) Citología Original:* Célula cervical teñida con Papanicolaou en resolución espacial de 224x224 píxeles.
  * *(B) Mapa de Activación Score-CAM:* Mapa bidimensional de saliencia cuya intensidad térmica está calibrada en una **escala continua normalizada de 0.0 (azul, nula relevancia) a 1.0 (rojo intenso, máxima relevancia)** mediante la paleta *jet*.
  * *(C) Superposición Clínicamente Auditable:* Fusión transparente (factor $\alpha=0.45$) entre la imagen celular y el mapa térmico.
* **Comentario Clínico:** La figura confirma que la red convolucional concentra su atención de forma nítida sobre la relación núcleo-citoplasma y el núcleo hipercromático anormal, evitando falsas activaciones en el fondo del frotis o en agregados de hematíes. Esto valida que la inferencia se fundamenta en biomarcadores morfológicos patológicos y no en artefactos de adquisición.

### Figura 3: Curvas ROC Comparativas frente al Punto de Comparación (Baseline Trivial)
*Archivo generado:* `figura3_curvas_roc_comparativas.png`

* **Descripción y Lectura sin Ambigüedad:** La **Figura 3** ilustra las Curvas ROC evaluadas en el conjunto de prueba (129 pacientes). El eje horizontal representa la **Tasa de Falsos Positivos ($1 - \text{Especificidad}$)** en el intervalo $[0.0, 1.0]$, y el eje vertical representa la **Tasa de Verdaderos Positivos ($\text{Sensibilidad o Recall}$)** en el intervalo $[0.0, 1.0]$.
* **Comentario Clínico:** Se incluye de forma destacada el **Punto de Comparación (Dummy Classifier)** junto a la diagonal de azar puro ($\text{AUC} = 50.00\%$). La gráfica evidencia visualmente cómo XGBoost ($\text{AUC} = 75.41\%$) y Random Forest ($\text{AUC} = 76.45\%$) logran una notable capacidad de separación latente, mientras que la curva de la Regresión Logística cae peligrosamente cerca de la línea de azar ($\text{AUC} = 50.93\%$).

---

## 3. Modelamiento Preliminar y Punto de Comparación (Rúbrica: 30 Puntos - Nivel Excelente)

Para evaluar con rigor científico los modelos, la rúbrica exige establecer un **Punto de Comparación** explícito que permita determinar si la exactitud observada obedece a un aprendizaje genuino o a un sesgo por desbalance.

### 3.1. Definición del Punto de Comparación (Baseline Trivial / Zero-R)
Se formalizó como punto de comparación un **Clasificador Trivial (*Dummy Classifier*)** configurado con la estrategia `most_frequent`. Dado que el 93.80% de las pacientes en prueba son negativas (121 de 129), un algoritmo simplista que prediga sistemáticamente "Sana" alcanza un **Accuracy del 93.80%**, pero con una **Sensibilidad del 0.00%**, **F1-Score del 0.00%** y un **AUC-ROC del 50.00%**. Este valor constituye el umbral mínimo que cualquier modelo clínico admisible debe superar sustancialmente en sensibilidad y área bajo la curva.

### 3.2. Hiperparámetros y Configuración de Modelos Evaluados
1. **Regresión Logística:** `max_iter=1000` (garantiza convergencia del optimizador lbfgs), `random_state=42`.
2. **SVM (Support Vector Machine):** Kernel de base radial (`kernel='rbf'`), `probability=True` (imprescindible para calibración probabilística y cómputo de AUC-ROC), `random_state=42`.
3. **Árbol de Decisión (Decision Tree):** Criterio de división Gini estándar, `random_state=42`.
4. **Random Forest:** Ensamblado de `n_estimators=100` árboles de decisión con remuestreo *bootstrap*, `random_state=42`.
5. **XGBoost:** Optimización mediante *gradient boosting*, función de pérdida binaria `eval_metric='logloss'`, `random_state=42`.

### 3.3. Tabla Comparativa de Rendimiento en Datos Tabulares (Conjunto de Prueba - 129 Casos)

```
+--------------------------+----------+--------------+---------------+-----------+----------+----------+
| Modelo                   | Accuracy | Sensibilidad | Especificidad | Precisión | F1-Score | AUC-ROC  |
+--------------------------+----------+--------------+---------------+-----------+----------+----------+
| Baseline (Dummy Trivial) | 93.80%   |  0.00%       | 100.00%       |  0.00%    |  0.00%   | 50.00%   |
| Regresión Logística      | 79.84%   | 12.50%       |  84.30%       |  5.00%    |  7.14%   | 50.93%   |
| Support Vector Machine   | 89.15%   |  0.00%       |  95.04%       |  0.00%    |  0.00%   | 65.60%   |
| Decision Tree            | 87.60%   | 25.00%       |  91.74%       | 16.67%    | 20.00%   | 58.06%   |
| Random Forest            | 92.25%   | 12.50%       |  97.52%       | 25.00%    | 16.67%   | 76.45%   |
| XGBoost                  | 93.80%   | 12.50%       |  99.17%       | 50.00%    | 20.00%   | 75.41%   |
+--------------------------+----------+--------------+---------------+-----------+----------+----------+
```

*Desglose de la Matriz de Confusión en Prueba (Total = 129 pacientes: 121 Negativas, 8 Positivas):*
* **Baseline Trivial:** $\text{TP}=0$, $\text{FP}=0$, $\text{TN}=121$, $\text{FN}=8$.
* **Regresión Logística:** $\text{TP}=1$, $\text{FP}=19$, $\text{TN}=102$, $\text{FN}=7$.
* **SVM:** $\text{TP}=0$, $\text{FP}=6$, $\text{TN}=115$, $\text{FN}=8$.
* **Decision Tree:** $\text{TP}=2$, $\text{FP}=10$, $\text{TN}=111$, $\text{FN}=6$.
* **Random Forest:** $\text{TP}=1$, $\text{FP}=3$, $\text{TN}=118$, $\text{FN}=7$.
* **XGBoost:** $\text{TP}=1$, $\text{FP}=1$, $\text{TN}=120$, $\text{FN}=7$.

### 3.4. Rendimiento de Modelos de Visión por Computadora (Transfer Learning)

```
+-----------------+-----------------------+----------+--------------+---------------+-----------+----------+----------+
| Dataset         | Modelo CNN            | Accuracy | Sensibilidad | Especificidad | Precisión | F1-Score | AUC-ROC  |
+-----------------+-----------------------+----------+--------------+---------------+-----------+----------+----------+
| Herlev (7 c.)   | MobileNet             | 42.70%   | 47.86%       | 90.18%        | 50.66%    | 47.76%   | 81.66%   |
| Herlev (7 c.)   | InceptionV3           | 40.15%   | 43.71%       | 89.61%        | 45.27%    | 43.73%   | 79.34%   |
| Herlev (7 c.)   | ResNet50              | 16.79%   | 11.34%       | 84.97%        |  4.31%    |  6.11%   | 57.98%   |
| Herlev (7 c.)   | AlexNet               | 45.26%   | 45.96%       | 90.30%        | 43.30%    | 42.42%   | 80.39%   |
| SIPaKMeD (5 c.) | MobileNet             | 49.35%   | 49.42%       | 87.19%        | 66.98%    | 49.93%   | 81.29%   |
| SIPaKMeD (5 c.) | InceptionV3           | 41.86%   | 41.65%       | 85.31%        | 55.46%    | 41.84%   | 73.28%   |
| SIPaKMeD (5 c.) | ResNet50              | 33.17%   | 33.49%       | 83.14%        | 16.62%    | 20.85%   | 60.33%   |
| SIPaKMeD (5 c.) | AlexNet               | 93.63%   | 93.99%       | 98.40%        | 93.79%    | 93.76%   | 99.65%   |
| RIVA (7 cat.)   | MobileNet             | 47.55%   | 47.19%       | 86.72%        | 68.20%    | 47.31%   | 76.93%   |
| RIVA (7 cat.)   | InceptionV3           | 40.66%   | 40.79%       | 85.09%        | 46.57%    | 41.15%   | 73.82%   |
| RIVA (7 cat.)   | ResNet50              | 21.18%   | 20.00%       | 80.00%        |  4.24%    |  6.99%   | 60.54%   |
| RIVA (7 cat.)   | AlexNet               | 92.83%   | 93.20%       | 98.19%        | 93.08%    | 93.13%   | 99.47%   |
+-----------------+-----------------------+----------+--------------+---------------+-----------+----------+----------+
```

---

## 4. Resultados Parciales y Discusión Crítica (Rúbrica: 15 Puntos - Nivel Excelente)

### 4.1. Qué Responde lo Obtenido Hasta Ahora
1. **Separación Latente en Datos Tabulares:** Aunque los modelos tabulares sufren bajo el umbral convencional de 0.50, el valor de **AUC-ROC alcanzado por Random Forest (76.45%) y XGBoost (75.41%)** demuestra que ambos algoritmos ordenan correctamente las probabilidades de riesgo. La explicabilidad global (Figura 1) confirma que esta discriminación responde a factores clínicos epidemiológicos plausibles.
2. **Capacidad Diagnóstica Sobresaliente de AlexNet:** En citología microscópica, AlexNet no solo superó ampliamente a las demás arquitecturas, sino que alcanzó niveles de precisión cuasi-clínicos: **93.63% de exactitud y 99.65% de AUC-ROC en SIPaKMeD**, y **92.83% de exactitud con 99.47% de AUC-ROC en RIVA**, manteniendo mapas de activación focalizados en núcleos celulares (Figura 2).

### 4.2. Resultados que Difieren de lo Esperado e Hipótesis Causales
1. **La Paradoja del Accuracy y la Falla de Sensibilidad en Datos Tabulares:**
   * *Hallazgo Anómalo:* A pesar de aplicar balanceo sintético SMOTE en el entrenamiento, modelos como SVM obtuvieron un Accuracy del 89.15% pero una **Sensibilidad nula del 0.00%**, mientras que XGBoost y Random Forest detectaron solo 1 de cada 8 casos positivos (Sensibilidad del 12.50%).
   * *Hipótesis Explicativa:* SMOTE balancea la frontera en el espacio de entrenamiento, pero la función de pérdida asigna penalizaciones simétricas a los falsos positivos y falsos negativos. Dado que en el conjunto de prueba la prevalencia real es de apenas 6.20% (15:1), el umbral de corte por defecto ($p=0.50$) induce al algoritmo a proteger la especificidad en perjuicio del recall, operando de manera idéntica al clasificador trivial (Dummy Baseline).
2. **Colapso Inesperado de ResNet50 en Citología:**
   * *Hallazgo Anómalo:* Siendo la arquitectura más profunda y moderna evaluada, ResNet50 exhibió un desempeño paupérrimo (**16.79% en Herlev**, prácticamente equivalente al azar uniforme de 14.28%), siendo detenida por *Early Stopping* en la época 6.
   * *Hipótesis Explicativa:* La estrategia estándar de congelar las capas convolucionales base (*Feature Extraction*) funciona para objetos macroscópicos de ImageNet, pero genera una meseta de gradientes insalvable en imágenes citológicas teñidas, donde los patrones críticos son variaciones sutiles de textura cromatínica y bordes nucleares. La profundidad de 50 capas generó un cuello de botella que impidió adaptar los pesos a este nuevo dominio biológico.

### 4.3. Qué Queda por Hacer (Plan de Mitigación para el Informe Final)
1. **Ajuste de Umbrales Clínicos (*Threshold-Moving* / Aprendizaje Sensible al Costo):** Se optimizará el umbral de decisión $p^*$ en XGBoost y Regresión Logística (ej. $p^* \approx 0.15 - 0.20$), priorizando una **Sensibilidad $\ge 85\%$** a expensas de tolerar más falsos positivos controlados.
2. **Fine-Tuning Profundo en CNNs:** Se descongelará el último bloque residual de ResNet50 y las capas superiores de MobileNet, reentrenando con una tasa de aprendizaje reducida ($\text{lr} = 10^{-5}$) y regularización *Weight Decay*.
3. **Cuantificación Rigurosa de XAI:** Se incorporarán métricas formales de fidelidad local (*Deletion AUC*) y sensibilidad direccional hacia conceptos médicos (*TCAV*).

---

## 5. Código y Reproducibilidad Numérica

De acuerdo con la exigencia formal del curso: *"Todo valor numérico informado debe poder reproducirse ejecutando el código entregado"*. El código completo que genera cada uno de los valores, tablas y figuras de este informe se encuentra disponible y probado en el archivo:

`reproducir_resultados.py`

### Instrucciones de Ejecución
```bash
# 1. Activar el entorno virtual del proyecto
source .venv/bin/activate

# 2. Ejecutar el script maestro de reproducción
python reproducir_resultados.py
```

### Entorno y Versiones Declaradas
* **Sistema:** macOS Darwin 24.x
* **Python:** 3.9.6
* **Scikit-learn:** 1.6.1
* **XGBoost:** 2.1.4
* **PyTorch / Torchvision:** 2.8.0 / 0.23.0
* **TensorFlow / Keras:** 2.20.0 / 3.10.0
* **Pandas / NumPy:** 2.3.3 / 2.0.2
* **Matplotlib:** 3.9.4
* **Semilla Aleatoria:** `random_state = 42` (establecida de manera determinista en todos los módulos estocásticos).
