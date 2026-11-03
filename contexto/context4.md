# Contexto 4: Actualización de Valores y Tablas de Métricas en Archivos Markdown (.md)
**Fecha de registro:** 2026-11-03

## 1. Instrucciones
* Actualizar todos los archivos `.md` del proyecto donde se hace mención a las tablas de evaluación y métricas de los modelos tras el nuevo entrenamiento.
* Reemplazar los valores antiguos por los valores actualizados consolidados en `image_metrics.json`, `metricas_reproducidas.csv` y `atribucion_shap_xgboost.csv`.
* Mantener total consistencia entre `informe_avance.md`, `analisis_resultados.md`, `resultados_entrenamiento.md`, `presentacion_oncopredict_hospital.md` y `README.md`.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Sincronizar la totalidad de las tablas y referencias numéricas en la documentación Markdown del proyecto con los resultados vigentes del último entrenamiento (datos tabulares UCI, citología Herlev/SIPaKMeD/RIVA en 8 clases Bethesda y colposcopía Intel & MobileODT).
* **Fases/Hitos principales:**
  1. Auditoría de valores actuales en `metricas_reproducidas.csv`, `image_metrics.json` y `atribucion_shap_xgboost.csv`.
  2. Actualización de las tablas y análisis en `informe_avance.md`, `analisis_resultados.md` y `resultados_entrenamiento.md`.
  3. Actualización de las cifras y tablas de resumen en `presentacion_oncopredict_hospital.md`, `README.md` y `reproducir_resultados.py`.
* **Entregables:**
  - `informe_avance.md` actualizado con las 16 configuraciones de Visión por Computadora y métricas vigentes.
  - `analisis_resultados.md` actualizado con tablas de UCI, SHAP y Visión por Computadora.
  - `resultados_entrenamiento.md`, `presentacion_oncopredict_hospital.md` y `README.md` sincronizados con las nuevas métricas.

## 3. Referencias
* **Path del plan original:** `informe_avance.md` y `analisis_resultados.md`
