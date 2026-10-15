# Contexto 2: Resolución del Punto 2 (Informe de Avance 20%) y Cumplimiento Estricto de la Rúbrica de Código
**Fecha de registro:** 2026-10-15

## 1. Instrucciones
* Cumplir estrictamente la exigencia de Código del archivo `CD_Proyecto.pdf`:
  - "Código. Cada informe se acompaña del código que produjo sus resultados. Su estilo no se evalúa; se exige que esté completo y que se ejecute. Todo valor numérico informado debe poder reproducirse ejecutando el código entregado, y los resultados que dependan de código que no se ejecuta se consideran no presentados."
* Auditar y reestructurar el "Informe de Avance: Proyecto de Ciencia de Datos (20%)" según las 4 dimensiones de la Rúbrica oficial para alcanzar la categoría "Excelente" (100/100 puntos):
  1. **Preprocesamiento (30 pts - Excelente):** Documentar estado inicial, cuantificar exhaustivamente registros afectados por datos ausentes, justificar SimpleImputer (mediana) frente al descarte por lista, explicitar exclusión por Data Leakage (3 columnas), partición estratificada 85/15 y balanceo SMOTE en entrenamiento.
  2. **Visualizaciones (25 pts - Excelente):** Incorporar y comentar formalmente tres figuras con ejes, escalas, unidades y función clínica (Figura 1: Atribución global de factores de riesgo; Figura 2: Explicabilidad visual local Grad-CAM/Score-CAM; Figura 3: Curvas ROC comparativas).
  3. **Modelamiento (30 pts - Excelente):** Incluir un punto de comparación formal (Clasificador Dummy/Trivial que predice siempre la clase mayoritaria) junto a los 5 modelos clásicos y los 4 modelos de Transfer Learning en imágenes.
  4. **Resultados (15 pts - Excelente):** Establecer con precisión qué responde y qué no responde lo obtenido, reportar resultados que difieren de lo esperado (paradoja del accuracy y colapso de ResNet50 en época 6) con sus hipótesis explicativas, y definir las líneas de trabajo pendientes (threshold-moving y fine-tuning profundo).
* Garantizar reproducibilidad determinista (`random_state=42`) mediante un script maestro autónomo (`reproducir_resultados.py`).

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Alinear el código y el texto del Informe de Avance (Entrega 2, 20%) con el estándar de máxima calificación de la rúbrica de Ciencia de Datos y el requisito absoluto de reproducibilidad numérica.
* **Fases/Hitos principales:**
  1. Auditoría cuantitativa del pipeline de datos e implementación del script maestro `reproducir_resultados.py`.
  2. Incorporación del punto de comparación (Dummy Classifier) y generación de las figuras analíticas de publicación.
  3. Redacción del informe final expandido y riguroso con la totalidad de los valores numéricos calculados por el código.
* **Entregables:**
  - Script ejecutable autónomo `reproducir_resultados.py`.
  - Figuras generadas: `figura1_importancia_atributos.png`, `figura2_explicabilidad_visual_xai.png`, `figura3_curvas_roc_comparativas.png`.
  - Archivo de métricas reproducidas: `metricas_reproducidas.csv`.
  - Informe de Avance reescrito y perfeccionado para entrega oficial.

## 3. Referencias
* **Path del plan original:** `CD_Proyecto.pdf`
