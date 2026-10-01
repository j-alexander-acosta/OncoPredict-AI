Actúa como un Ingeniero de Machine Learning Senior especializado en Visión por Computadora Médica e Inteligencia Artificial Explicable (XAI). Actualmente, estamos desarrollando el proyecto "OncoPredict AI", utilizando EfficientNet-B0 en PyTorch para clasificar imágenes de citología (datasets SIPaKMeD y RIVA) y preparándonos para analizar imágenes macroscópicas de colposcopía.

Necesito que escribas el código en Python (PyTorch) para elevar los niveles de explicabilidad de nuestro modelo, pasando de un enfoque cualitativo (visual) a uno estrictamente cuantitativo y auditable.

Estructura tu respuesta en cuatro scripts o módulos claramente definidos, cumpliendo los siguientes requerimientos:

Módulo 1: Interpretabilidad de Alta Resolución (Reemplazo de Grad-CAM)

Escribe una clase robusta que implemente Score-CAM (o HiResCAM).

El objetivo es generar mapas de activación de alta fidelidad que no dependan de los gradientes (para evitar el ruido visual), delineando con precisión las estructuras celulares en las imágenes de entrada de 224x224.

Módulo 2: Explicabilidad Cuantitativa (Insertion / Deletion AUC)

Escribe una función de evaluación algorítmica que tome las imágenes originales, las predicciones del modelo y los mapas de calor generados en el Módulo 1.

La función debe implementar la métrica de Deletion AUC (oscureciendo progresivamente los píxeles más importantes según el mapa de calor y midiendo la caída de la probabilidad de la clase predicha).

La función debe devolver un valor numérico exacto (AUC) que cuantifique la fidelidad del mapa de calor, permitiéndonos reportar matemáticamente si el modelo realmente está mirando la lesión.

Módulo 3: Explicabilidad Global (Bases para TCAV)

Escribe el código fundacional para implementar TCAV (Testing with Concept Activation Vectors).

Necesito una función que extraiga las activaciones de una capa profunda específica de EfficientNet-B0 para un conjunto de imágenes de un "concepto médico" (ej. zonas acetoblancas).

Escribe la lógica para entrenar un clasificador lineal (SVM) sobre estas activaciones y calcular la Sensibilidad Conceptual, obteniendo una métrica global (ej. "El concepto X contribuye al 85% de las predicciones").

Módulo 4: Reporte y Tabla Comparativa de Explicabilidad

Escribe una función que genere e imprima (en formato Markdown o DataFrame de Pandas) una tabla idéntica a la "Tabla 3. Nivel de explicabilidad de los algoritmos evaluados" bajo el título "5.2. Evaluación Comparativa de Explicabilidad".   
PNG

La tabla debe contener exactamente estas cinco columnas: Algoritmo, Transparencia, Interpretabilidad, Auditabilidad, y Explicabilidad Global.   
PNG

Incluye las filas de los modelos tradicionales (Regresión Logística, Decision Tree, SVM, Random Forest, XGBoost) con sus métricas originales, pero actualiza la fila de la CNN para que refleje la nueva arquitectura (ej. CNN (EfficientNet-B0 + Score-CAM + TCAV)), demostrando con variables cómo sus niveles subieron a "Media-Alta" o "Alta" gracias a las técnicas implementadas.   
PNG

Restricciones de Código:

El código debe ser modular y estar listo para integrarse con nuestro pipeline actual (que usa torchvision.transforms y DataLoader).

Incluye comentarios técnicos detallados explicando la matemática detrás de cada paso.

Asegura compatibilidad con aceleración por hardware (MPS para Apple Silicon y CUDA).
