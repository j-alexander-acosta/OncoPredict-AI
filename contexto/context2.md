# Contexto 2: Integración y Reentrenamiento del Dataset Oficial RIVA (8 Categorías Bethesda)
**Fecha de registro:** 2026-11-02

## 1. Instrucciones
* Reemplazar la copia duplicada previa de SIPaKMeD en `Media/RIVA/` por el dataset oficial **RIVA v1.0** (*An Image Dataset of Conventional Pap Smear Cytology with Multiple Independent Annotations*, Zenodo `riva_1.0`).
* Estructurar `Media/RIVA/` en carpetas por clase siguiendo la arquitectura híbrida de frotis completos + recortes celulares (`CROPPED/`) a partir de `annotations.json`.
* Reentrenar los 5 modelos de Visión por Computadora sobre el nuevo dataset RIVA:
  1. `EfficientNet-B0` (PyTorch / Fine-Tuning Progresivo)
  2. `AlexNet` (PyTorch)
  3. `MobileNet` (TensorFlow / Keras)
  4. `InceptionV3` (TensorFlow / Keras)
  5. `ResNet50` (TensorFlow / Keras)
* Actualizar los pesos en `Media/models/*_riva.*`, el archivo de métricas `image_metrics.json`, las plantillas web (`templates/index.html`) y la documentación del proyecto (`README.md`, `resultados_entrenamiento.md`).

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Incorporar el dataset citológico real RIVA (959 frotis completos y 15,882 células únicas recortadas en 8 categorías del Sistema Bethesda) y actualizar el benchmark comparativo de modelos CNN.
* **Fases/Hitos principales:**
  1. Procesamiento de `annotations.json` con agrupamiento espacial de anotadores múltiples y generación de 15,882 parches celulares de `100x100` píxeles junto a los 959 frotis en 8 carpetas de clases (`ASCH`, `ASCUS`, `ENDO`, `HSIL`, `INFL`, `LSIL`, `NILM`, `SCC`).
  2. Adaptación de `infer_pytorch.py`, `train_missing.py` y `templates/index.html` para soportar dinámicamente las 8 clases de RIVA.
  3. Reentrenamiento de las 5 arquitecturas CNN sobre RIVA y actualización de `image_metrics.json` y documentación técnica.
* **Entregables:**
  - Directorio `Media/RIVA/` estructurado con 16,841 imágenes en 8 clases Bethesda.
  - Modelos entrenados (`efficientnet_b0_riva.pth`, `alexnet_riva.pth`, `mobilenet_riva.keras`, `inceptionv3_riva.keras`, `resnet50_riva.keras`).
  - `image_metrics.json` y reportes actualizados.

## 3. Referencias
* **Path del plan original:** `README.md` y `train_missing.py`
