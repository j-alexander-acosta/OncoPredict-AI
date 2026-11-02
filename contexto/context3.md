# Contexto 3: Implementación de SHAP sobre XGBoost y Incorporación de Tablas XAI en Entregables
**Fecha de registro:** 2026-11-02

## 1. Instrucciones
* Calcular los valores SHAP reales (`mean(|SHAP value|)` en *log-odds*) del modelo `XGBoost` sobre el dataset clínico tabular UCI dentro de `reproducir_resultados.py`.
* Agregar la **Tabla de Atribución SHAP (Top Factores de Riesgo en XGBoost)** y la **Tabla Resumen de Técnicas XAI (Evaluación Cualitativa y Post-Hoc)** en:
  - `Resumen_Configuracion_e_Hiperparametros.xlsx` (mediante `generate_excel_summary.py`)
  - `Resumen_Configuracion_e_Hiperparametros.docx` (mediante `generate_word_summary.py`)
  - `informe_avance.md`
* Asegurar total coherencia con la Diapositiva 11 de la presentación `PPT-INFONOR2026` (SHAP en Datos Tabulares con XGBoost y Grad-CAM/Score-CAM en Imágenes con AlexNet).

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar el cálculo determinista de valores TreeSHAP en `XGBoost` (unidades *log-odds*) y reflejar sus resultados tabulares y el resumen XAI en todos los documentos y tablas entregables del proyecto.
* **Fases/Hitos principales:**
  1. Cómputo de TreeSHAP (`mean(|SHAP|)` en *log-odds*, dirección de efecto y porcentaje de atribución) para `XGBoost` en `reproducir_resultados.py` y `model_service.py`, actualizando también la visualización de importancia/SHAP.
  2. Actualización de `generate_excel_summary.py` y `generate_word_summary.py` para incluir las hojas/tablas `Atribucion_SHAP_XGBoost` y `Resumen_Tecnicas_XAI`.
  3. Actualización de `informe_avance.md` con la tabla numérica de valores SHAP de XGBoost junto a la atribución global y la tabla resumen de validación post-hoc XAI.
* **Entregables:**
  - `reproducir_resultados.py` actualizado y ejecutado.
  - `Resumen_Configuracion_e_Hiperparametros.xlsx` y `Resumen_Configuracion_e_Hiperparametros.docx` regenerados con las nuevas tablas.
  - `informe_avance.md` actualizado con las tablas de SHAP y XAI.

## 3. Referencias
* **Path del plan original:** `informe_avance.md`
