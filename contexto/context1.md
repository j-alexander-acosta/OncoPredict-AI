# Contexto 1: Implementación de Módulos XAI en OncoPredict AI
**Fecha de registro:** 2026-10-06

## 1. Instrucciones
* Elevar los niveles de explicabilidad del modelo EfficientNet-B0.
* Pasar de un enfoque cualitativo visual a uno cuantitativo y auditable.
* Implementar 5 módulos de explicabilidad en Python (PyTorch):
  1. Interpretabilidad de Alta Resolución (Score-CAM).
  2. Explicabilidad Cuantitativa (Insertion / Deletion AUC).
  3. Explicabilidad Global (Bases para TCAV).
  4. Reporte y Tabla Comparativa de Explicabilidad.
  5. Documento Teórico - Estado del Arte en XAI.
* Asegurar que el código sea modular, documentado técnicamente y compatible con aceleración por hardware (MPS para Apple Silicon y CUDA).

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar un sistema avanzado de Inteligencia Artificial Explicable (XAI) cuantificable para respaldar clínicamente el análisis de imágenes de citología y colposcopía.
* **Fases/Hitos principales:**
  1. Desarrollo de los scripts técnicos (Módulos 1 a 4) en el directorio `xai_modules/`.
  2. Generación del reporte de evaluación comparativa de modelos.
  3. Redacción del documento teórico del estado del arte justificando las técnicas elegidas (Módulo 5).
* **Entregables:** 
  - Scripts PyTorch (`score_cam.py`, `quantitative_metrics.py`, `tcav_core.py`, `explainability_report.py`).
  - Tabla de métricas XAI generada.
  - Ensayo académico `documento_teorico_xai.md`.

## 3. Referencias
* **Path del plan original:** `propuesta_para_incrementar_la_explicabilidad.md`
