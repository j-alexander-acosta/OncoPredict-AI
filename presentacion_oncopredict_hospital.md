# Presentación OncoPredict AI - Hospital Regional de Antofagasta

---

### **Diapositiva 1: Título y Presentación**

**Texto visible (Bullet points):**
* **OncoPredict AI:** Detección Temprana de Cáncer Cervicouterino mediante Machine Learning e Inteligencia Artificial Explicable (XAI).
* **Institución:** Doctorado en Inteligencia Artificial, Universidad Católica del Norte.
* **Expositor:** Alexander Acosta, estudiante de Doctorado de Inteligencia Artificial, Universidad Católica del Norte.

**Notas del orador (Guion detallado):**
> "Muy buenos días a la dirección y al honorable equipo médico del Hospital Regional de Antofagasta. Es un honor estar hoy con ustedes. Mi nombre es Alexander Acosta, y represento al programa de Doctorado en Inteligencia Artificial de la Universidad Católica del Norte. Hoy vengo a presentarles 'OncoPredict AI', un proyecto de investigación aplicada que busca revolucionar la detección temprana del cáncer cervicouterino. Nuestro objetivo principal en esta sesión es mostrarles los últimos avances que hemos logrado integrando Machine Learning y, lo que es más importante, proponerles una colaboración estratégica y tecnológica que beneficie directamente a este hospital y a las pacientes de la macrozona norte."

---

### **Diapositiva 2: El Problema Clínico y la Opacidad de la IA**

**Texto visible (Bullet points):**
* **Limitaciones actuales:** El tamizaje tradicional (Papanicolaou) presenta variabilidad inter-observador y alta dependencia de personal ultra-especializado.
* **El desafío tecnológico (La "Caja Negra"):** Los modelos tradicionales de IA médica carecen de transparencia en sus decisiones.
* **Nuestra solución:** Integración de Inteligencia Artificial Explicable (XAI) para auditar predicciones y generar verdadera confianza clínica.

**Notas del orador (Guion detallado):**
> "Como ustedes bien saben, el estándar actual de tamizaje, como el Papanicolaou, es fundamental, pero no está exento de limitaciones. Depende en gran medida de la disponibilidad de citopatólogos altamente especializados y está sujeto a variabilidad humana. Para mitigar esto, la IA ha prometido soluciones, pero ha chocado con un muro: el problema de la 'Caja Negra'. Los médicos, con justa razón, no pueden confiar en un diagnóstico emitido por un algoritmo si no entienden *por qué* tomó esa decisión. OncoPredict AI aborda este problema de frente integrando Inteligencia Artificial Explicable, o XAI. No solo buscamos predicciones precisas, sino modelos auditables que justifiquen clínicamente sus hallazgos, construyendo así una herramienta de apoyo, no un reemplazo, en la que el profesional de la salud pueda confiar."

---

### **Diapositiva 3: Arquitectura Triple del Proyecto**

**Texto visible (Bullet points):**
* **Enfoque Integral:** Predicción y análisis desde tres frentes complementarios:
    1. **Inferencia Clínica:** Análisis de Factores de Riesgo (Datos Tabulares).
    2. **Visión por Computadora Microscópica:** Análisis Celular (Citología Digital).
    3. **Visión por Computadora Macroscópica:** Análisis de Tejido In-vivo (Colposcopía).

**Notas del orador (Guion detallado):**
> "Para abordar una patología tan compleja, un solo enfoque no basta. Por ello, el proyecto ha evolucionado hacia una arquitectura triple. Primero, analizamos el perfil del paciente mediante Inferencia Clínica basada en datos tabulares de factores de riesgo. Segundo, bajamos al nivel celular mediante Visión por Computadora aplicada a la Citología Digital. Y tercero, abordamos el escenario clínico directo: el análisis macroscópico in-vivo a través de imágenes colposcópicas. Esta triangulación nos permite tener una visión predictiva holística, desde la historia clínica hasta el tejido vivo."

---

### **Diapositiva 4: Fase 1 - Inferencia Clínica (Datos Tabulares)**

**Texto visible (Bullet points):**
* **Dataset:** UCI Cervical Cancer Risk Factors (858 pacientes, 36 atributos médicos/conductuales).
* **Modelos Evaluados:** Regresión Logística, SVM, Árboles de Decisión, Random Forest, XGBoost.
* **Técnicas de Optimización:** Scikit-learn, Pandas, SMOTE para balanceo de clases.
* **Hallazgos Clave:** 
    * XGBoost logra ~93.8% de exactitud (métrica engañosa por desbalance).
    * Importancia crítica del ajuste del umbral clínico (*threshold*) para maximizar la sensibilidad y detectar verdaderos positivos.

**Notas del orador (Guion detallado):**
> "En nuestra primera fase, analizamos datos tabulares utilizando el dataset de la UCI con 858 pacientes y 36 atributos. Evaluamos varios algoritmos robustos y, aunque modelos como XGBoost alcanzaron una exactitud aparente del 93.8%, el análisis profundo reveló que esta métrica es engañosa debido al desbalance natural de los datos clínicos. Lo verdaderamente importante aquí es la sensibilidad. A través de técnicas como SMOTE y el ajuste dinámico del umbral clínico (*threshold*), logramos reconfigurar el modelo para que priorice la detección de los verdaderos positivos. En oncología, un falso negativo es inaceptable, y nuestras herramientas de explicabilidad nos permitieron ajustar el algoritmo precisamente para este requerimiento médico."

---

### **Diapositiva 5: Fase 2 - Análisis Microscópico (Citología Digital)**

**Texto visible (Bullet points):**
* **Datasets:** SIPaKMeD (~21,200 img), RIVA (~21,200 img), Herlev (~900 img). *Aviso: Actualmente no existe un dataset chileno para imágenes microscópicas.*
* **Infraestructura:** PyTorch con aceleración nativa MPS (Apple Silicon).
* **Innovación Tecnológica (SOTA):** Superación del "estancamiento morfológico" mediante **EfficientNet-B0** y estrategias de **Fine-Tuning Progresivo** (descongelamiento dinámico).
* **Resultados Campeones:** 
    * SIPaKMeD: Exactitud **94.42%** | AUC-ROC **99.39%**
    * RIVA: Exactitud **93.63%**

**Notas del orador (Guion detallado):**
> "Pasando a la fase celular, entrenamos nuestros modelos con más de 40.000 imágenes citológicas. Históricamente, arquitecturas previas sufrían de 'estancamiento morfológico celular', siendo incapaces de diferenciar anomalías sutiles. Resolvimos este problema adoptando la arquitectura de estado del arte, EfficientNet-B0, impulsada en PyTorch. Pero el verdadero salto de calidad se logró implementando una técnica de *Fine-Tuning* Progresivo, descongelando dinámicamente los bloques convolucionales de la red. Los resultados son sobresalientes: alcanzamos una exactitud del 94.42% y un AUC-ROC casi perfecto del 99.39% en el dataset SIPaKMeD. Estas métricas demuestran una capacidad de generalización excepcional. Sin embargo, es vital destacar un punto crítico: a pesar del volumen de datos, **ninguno de estos datasets microscópicos es chileno**. Esta carencia refuerza nuestra necesidad urgente de generar datos locales a nivel de nuestro país, algo que da pie y fundamenta la propuesta colaborativa que les presentaremos."

---

### **Diapositiva 6: Fase 3 - Análisis Macroscópico (Colposcopía In-Vivo)**

**Texto visible (Bullet points):**
* **Dataset Base:** Intel & MobileODT Cervical Cancer Screening.
* **Objetivo Clínico:** Clasificación anatómica de la Zona de Transformación in-vivo (Tipos 1, 2 y 3).
* **Arquitectura:** EfficientNet-B0 adaptado para imágenes colposcópicas.
* **El Puente:** Cerrando la brecha entre el laboratorio de patología y el box de atención ginecológica.

**Notas del orador (Guion detallado):**
> "La tercera fase nos lleva directamente al box de atención. Utilizando imágenes del desafío Intel y MobileODT, hemos adaptado nuestro modelo para el análisis in-vivo, con el objetivo de asistir en la clasificación del tipo de Zona de Transformación (Tipos 1, 2 y 3). Esta fase es crucial porque es el puente tecnológico que conecta el análisis microscópico de laboratorio con la realidad macroscópica que el ginecólogo observa a través del colposcopio. La meta es proveer una 'segunda opinión' instantánea durante el examen físico."

---

### **Diapositiva 7: Inteligencia Artificial Explicable (XAI) en Acción**

**Texto visible (Bullet points):**
* **De lo cualitativo a lo cuantitativo:** Transparencia total en la inferencia.
* **Análisis de Factores de Riesgo (SHAP):** Auditoría del impacto individual de variables clínicas (edad, paridad, ITS).
* **Visión por Computadora (Grad-CAM / LIME):** Mapas de calor (activación visual) que destacan áreas anómalas en células y tejido cervical.

**Notas del orador (Guion detallado):**
> "Como mencioné al inicio, la precisión sin confianza no sirve en medicina. Por eso implementamos Inteligencia Artificial Explicable de forma transversal. Pasamos de 'creerle a la máquina' a auditar matemáticamente sus decisiones. Para los datos clínicos, usamos la técnica SHAP, que nos dice exactamente qué peso tuvo cada factor de riesgo en la predicción. Para las imágenes, implementamos Grad-CAM y LIME, que generan mapas de calor sobre la citología o la colposcopía. Esto significa que la IA no solo dice 'sospecha de malignidad', sino que ilumina en la pantalla exactamente qué grupo celular o qué área del epitelio la llevó a esa conclusión, permitiendo al especialista validar el resultado visualmente."

---

### **Diapositiva 8: Consideraciones Éticas y Legales**

**Texto visible (Bullet points):**
* **Privacidad de Datos:** Cumplimiento estricto de la Ley N° 19.628 (Protección de la Vida Privada).
* **Anonimización Garantizada:** Eliminación absoluta de RUT e identificadores personales en todo flujo de datos.
* **Respaldo Institucional:** Sometimiento riguroso del protocolo a los Comités Ético Científicos de la Universidad Católica del Norte y del Hospital Regional, además de la revisión por el Comité Jurídico del hospital.

**Notas del orador (Guion detallado):**
> "Entendemos profundamente que trabajamos con el activo más sensible: la información de los pacientes. El diseño de este proyecto garantiza el cumplimiento estricto de la Ley 19.628 de Protección de la Vida Privada. Nuestro pipeline de datos está diseñado con anonimización absoluta desde el origen; ningún algoritmo verá jamás un RUT o nombre. Para garantizar la total transparencia y viabilidad institucional, todo este marco procedimental y el protocolo de investigación serán sometidos a una revisión exhaustiva: por los Comités Ético Científicos tanto de la Universidad Católica del Norte como del Hospital Regional de Antofagasta, y por supuesto, por el Comité Jurídico del hospital para su validación final."

---

### **Diapositiva 9: La Propuesta Institucional para el Hospital Regional**

**Texto visible (Bullet points):**
* **Modernización de Equipamiento:** Digitalización del colposcopio **Karl Kaps SOM 52** (UPC).
    * Integración: Divisor de haz (*beam splitter*), cámara médica (Montura C) y pedal USB.
* **Flujo de Trabajo "Manos Libres":** Captura de imágenes sin interrumpir la atención.
* **Beneficio Mutuo - El Hito Tecnológico:** 
    * **Para el Hospital:** Un equipo permanentemente digitalizado a costo cero para la institución.
    * **Para OncoPredict AI:** Creación de un dataset colposcópico local inédito, validado con el *Ground Truth* de biopsias de Anatomía Patológica, para entrenar la primera IA multimodal chilena.

**Notas del orador (Guion detallado):**
> "Y esto nos lleva al núcleo de nuestra visita de hoy: una propuesta de colaboración donde ambas partes ganan enormemente. Proponemos modernizar el equipo de la Unidad de Patología Cervical, específicamente digitalizando el colposcopio Karl Kaps SOM 52. Proveeremos e instalaremos un divisor de haz, una cámara médica de alta resolución y un pedal de captura. Para el hospital y los médicos, esto significa un flujo de trabajo 'manos libres' y un equipo modernizado permanentemente. A cambio, el proyecto OncoPredict AI podrá comenzar a construir un dataset colposcópico local, absolutamente inédito en nuestra región. Al cruzar estas imágenes in-vivo con el *Ground truth*, es decir, los resultados de las biopsias de Anatomía Patológica, podremos entrenar la primera IA multimodal verdaderamente adaptada a la demografía chilena."

---

### **Diapositiva 10: Conclusiones y Próximos Pasos**

**Texto visible (Bullet points):**
* **Impacto Inmediato:** Optimización del tamizaje en la red pública de la macrozona norte.
* **Liderazgo Tecnológico:** Posicionar al Hospital Regional como pionero en adopción de IA explicable en oncología ginecológica.
* **Siguientes Pasos:** Evaluación técnica del equipo y revisión ética.
* **¿Preguntas?**

**Notas del orador (Guion detallado):**
> "En conclusión, esta colaboración no solo optimizará los procesos de tamizaje en nuestra red pública, reduciendo los tiempos de espera y mejorando la precisión diagnóstica, sino que posicionará al Hospital Regional de Antofagasta a la vanguardia nacional en la adopción ética y tecnológica de Inteligencia Artificial en oncología. Los siguientes pasos serían coordinar una visita técnica a la unidad para evaluar el equipo y comenzar los trámites formales con el Comité de Ética. Agradezco sinceramente su tiempo, visión y atención esta mañana. Quedo a su entera disposición para responder cualquier pregunta o profundizar en los detalles técnicos y clínicos de la propuesta."
