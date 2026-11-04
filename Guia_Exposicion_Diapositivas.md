# Guía de Exposición y Notas del Orador por Diapositiva
**Presentación:** `Presentacion_Analisis_Comparativo_CaCu_ML.pptx` (20 diapositivas)  
**Artículo:** *Análisis Comparativo de Métodos de Machine Learning para la Detección Temprana de Cáncer Cervicouterino: Avances, Brechas y Desafíos de Explicabilidad*  
**Autores:** Jacinto Alexander Acosta Zambrano y Claudio Meneses Villegas — Universidad Católica del Norte (Antofagasta, Chile)

---

## Diapositiva 1: Portada
* **Objetivo:** Presentar el título de la investigación, los autores, la filiación institucional y el marco de financiamiento del estudio.
* **Puntos clave que no debes pasar por alto:**
  1. **Enfoque dual del título:** Destacar que el trabajo no solo compara algoritmos de *Machine Learning* (ML) para el tamizaje temprano de cáncer cervicouterino (CaCu), sino que aborda críticamente las **brechas metodológicas** y los **desafíos de explicabilidad clínica (XAI)**.
  2. **Afiliación y contexto regional:** Mencionar el trabajo desarrollado en la Universidad Católica del Norte (Antofagasta, Chile) bajo el Proyecto **BIP 40074839-0** (*Formación de Capital Humano Avanzado en IA para la Revolución Digital de la Región de Antofagasta*).
* **Guion sugerido:**
  > «Buenos días/tardes. Presento el estudio titulado *"Análisis Comparativo de Métodos de Machine Learning para la Detección Temprana de Cáncer Cervicouterino: Avances, Brechas y Desafíos de Explicabilidad"*, desarrollado junto al Dr. Claudio Meneses en la Universidad Católica del Norte, en el marco del proyecto regional de Capital Humano Avanzado en IA (BIP 40074839-0). Nuestra investigación aborda cómo articular la precisión algorítmica con la interpretabilidad clínica necesaria para llevar estos modelos a la práctica médica real.»

---

## Diapositiva 2: Resumen Ejecutivo — Problemática y Hallazgos Clave
* **Objetivo:** Ofrecer una vista panorámica de toda la presentación en un minuto: por qué se hizo el estudio, cómo se diseñó y qué se descubrió.
* **Puntos clave que no debes pasar por alto:**
  1. **Limitación del Papanicolaou:** Su sensibilidad es variable y depende fuertemente de citopatólogos expertos.
  2. **Marco integrador tridimensional de XAI:** Evaluar no solo métricas predictivas, sino **transparencia, interpretabilidad y auditabilidad**.
  3. **Evaluación bimodal:** Comparación en paralelo de datos clínicos tabulares (factores de riesgo en UCI) e imágenes citológicas (CNN en Herlev, SIPaKMeD y RIVA).
  4. **Contraste de resultados:** En datos tabulares, XGBoost logra alta exactitud (`93.80%`) pero baja sensibilidad (`12.50%`) con el umbral por defecto debido al desbalance; en imágenes, **AlexNet** lidera con `93.63%` de exactitud en SIPaKMeD y `59.20%` en RIVA (8 clases).
  5. **Directriz clínica:** Es indispensable calibrar el umbral de decisión (*thresholding*) e integrar explicabilidad *post-hoc* (**SHAP** y **Grad-CAM**).
* **Guion sugerido:**
  > «A modo de resumen ejecutivo: partimos de las limitaciones del Papanicolaou tradicional y proponemos un marco que une el rendimiento predictivo con tres dimensiones de explicabilidad: transparencia, interpretabilidad y auditabilidad. Evaluamos dos modalidades: datos clínicos tabulares y visión por computadora en citología. Demostramos que en datos tabulares una alta exactitud puede ocultar una baja sensibilidad por el desbalance de clases, mientras que en citología redes como AlexNet alcanzan un desempeño sobresaliente. Concluimos que el despliegue clínico exige calibrar los umbrales de decisión y auditar las predicciones con SHAP y Grad-CAM.»

---

## Diapositiva 3: Indicadores Empíricos Destacados (KPIs)
* **Objetivo:** Fijar en la audiencia las 4 cifras clave que sostienen las conclusiones del artículo.
* **Puntos clave que no debes pasar por alto:**
  1. **`93.80%` (Exactitud XGBoost en UCI):** Explicar que viene acompañada de solo `12.50%` de sensibilidad con el corte estándar ($p = 0.50$), lo que ejemplifica la **«Paradoja del Accuracy»**.
  2. **`93.63%` (Exactitud AlexNet en SIPaKMeD):** Subrayar su **AUC-ROC de `99.65%`**, sensibilidad de `93.99%` y especificidad de `98.40%` en clasificación celular de 5 clases.
  3. **`59.20%` (Exactitud AlexNet en RIVA):** Resaltar que se trata de un problema complejo de **8 categorías del Sistema Bethesda** (donde el azar es `12.5%`), alcanzando un **AUC-ROC de `88.90%`** y especificidad de `93.49%`.
  4. **`< 5%` (Modelos con XAI validada):** Menos del 5% de los estudios publicados validan clínicamente sus métodos de explicabilidad con especialistas.
* **Guion sugerido:**
  > «Estos cuatro indicadores sintetizan nuestros hallazgos: primero, el 93,80% de exactitud de XGBoost en datos tabulares que, con apenas 12,50% de sensibilidad bajo el umbral estándar, ilustra el peligro de la "Paradoja del Accuracy". Segundo, en citología unicelular (SIPaKMeD), AlexNet alcanza 93,63% de exactitud y un AUC-ROC de 99,65%. Tercero, en frotis multicelulares complejos de 8 clases (RIVA), AlexNet logra 59,20% de exactitud y 88,90% de AUC-ROC. Y cuarto, una brecha crítica de la literatura: menos del 5% de los modelos publicados validan sus explicaciones XAI en entornos clínicos.»

---

## Diapositiva 4: Introducción — Magnitud Epidemiológica y Tamizaje
* **Objetivo:** Contextualizar la urgencia médica y de salud pública del cáncer cervicouterino a nivel global y en Chile.
* **Puntos clave que no debes pasar por alto:**
  1. **Carga mundial (GLOBOCAN):** Más de `600.000` nuevos casos y cerca de `340.000` muertes al año; el **90%** ocurre en países de ingresos bajos y medios.
  2. **Realidad en Chile (DEIS):** Incidencia de `9,1` por cada 100.000 mujeres y alrededor de `550` muertes anuales.
  3. **Etiología y ventana clínica:** Causado por la infección persistente del VPH de alto riesgo (genotipos 16 y 18), pero con una progresión lenta de **10 a 20 años**, lo que da una amplia ventana para la detección temprana.
  4. **Debilidades del Papanicolaou convencional:** Sensibilidad muy variable (`51%` a `95%`, promedio habitual `70%–80%`), alta dependencia de citopatólogos expertos, fatiga visual/variabilidad interobservador que eleva los falsos negativos, y **déficit territorial de especialistas** en zonas rurales y extremas de Chile.
* **Guion sugerido:**
  > «El cáncer cervicouterino representa más de 600.000 casos y 340.000 muertes anuales en el mundo, y unas 550 defunciones al año en Chile. Aunque la evolución desde la infección por VPH toma entre 10 y 20 años —dándonos una gran ventana de prevención—, el tamizaje mediante Papanicolaou presenta una sensibilidad variable (entre 51% y 95%), sufre por la fatiga visual humana y enfrenta un déficit crítico de citopatólogos en regiones apartadas.»

---

## Diapositiva 5: Introducción — Potencial del ML y Barreras Clínicas
* **Objetivo:** Contrastar las capacidades del *Machine Learning* frente a los obstáculos reales que impiden su adopción en hospitales.
* **Puntos clave que no debes pasar por alto:**
  1. **Oportunidades del ML/DL:**
     * Las **CNN** igualan a expertos en análisis citológico y colposcópico.
     * Los **modelos tabulares** (Random Forest, SVM, XGBoost) perfilan el riesgo clínico y demográfico.
     * Permiten **automatizar el triaje**, priorizando casos de alto riesgo y acortando tiempos diagnósticos.
  2. **Tres barreras críticas de adopción:**
     * **Efecto «Caja Negra»:** Sin explicaciones comprensibles, el médico no confía ni puede asumir responsabilidad legal sobre la predicción.
     * **Domain Shift (Desplazamiento de dominio):** Los modelos caen en rendimiento cuando pasan de un dataset limpio de laboratorio a muestras hospitalarias heterogéneas (diferentes tinciones, cámaras o poblaciones).
     * **Resiliencia operativa:** Limitaciones de hardware, conectividad y privacidad de datos en hospitales públicos, donde el **aprendizaje federado** surge como vía de solución.
* **Guion sugerido:**
  > «Frente a esto, el Machine Learning ofrece una sinergia clara: algoritmos clásicos para perfilar riesgo clínico y redes convolucionales para analizar citología, permitiendo priorizar pacientes de alto riesgo. Sin embargo, su adopción clínica choca con tres barreras: el efecto "caja negra" que frena la confianza médica, el *domain shift* cuando cambian las condiciones de tinción o captura entre hospitales, y las limitaciones de infraestructura y privacidad.»

---

## Diapositiva 6: Objetivos del Estudio
* **Objetivo:** Definir con precisión el objetivo general y los cuatro objetivos específicos que estructuran la investigación.
* **Puntos clave que no debes pasar por alto:**
  1. **Objetivo General:** Comparar sistemáticamente métodos de ML para la detección temprana de CaCu integrando **rendimiento predictivo** y **explicabilidad clínica (XAI)**.
  2. **Objetivo (i) — Marco Teórico:** Articular la biología de la enfermedad (Bethesda, OMS), los algoritmos ML/DL y la taxonomía XAI.
  3. **Objetivo (ii) — Brechas del Arte:** Identificar vacíos en validación cuantitativa de XAI y estandarización experimental.
  4. **Objetivo (iii) — Evaluación Empírica:** Ejecutar un *benchmark* unificado sobre datasets públicos (**UCI, Herlev, SIPaKMeD y RIVA**) con métricas comunes.
  5. **Objetivo (iv) — Auditoría XAI:** Evaluar transparencia, interpretabilidad y auditabilidad aplicando **SHAP** (tabular) y **Grad-CAM** (imágenes).
* **Guion sugerido:**
  > «Por ello planteamos como objetivo general comparar sistemáticamente métodos de Machine Learning integrando su capacidad predictiva con su explicabilidad clínica. Esto se desglosa en cuatro objetivos específicos: primero, construir un marco teórico integrador; segundo, identificar las brechas metodológicas del estado del arte; tercero, realizar una evaluación empírica estandarizada en cuatro repositorios públicos; y cuarto, auditar los modelos bajo las dimensiones de transparencia, interpretabilidad y auditabilidad.»

---

## Diapositiva 7: Marco Teórico — Biología Cervical y Modelos ML/DL
* **Objetivo:**Conectar la base médica (evolución histológica y Sistema Bethesda) con las familias de algoritmos empleadas.
* **Puntos clave que no debes pasar por alto:**
  1. **Historia natural y Sistema Bethesda (Sección 2.1):**
     * Progresión desde infección persistente por VPH hacia neoplasia intraepitelial cervical (NIC 1, 2, 3) y carcinoma invasor.
     * El **Sistema Bethesda** estandariza la citología en **NILM** (negativo), **ASC-US** (atipia de significado indeterminado), **LSIL** (lesión de bajo grado) y **HSIL** (lesión de alto grado), que son precisamente las etiquetas de datasets modernos como **RIVA**.
     * Alineación con la estrategia 90-70-90 de la **OMS** para erradicar el CaCu.
  2. **Algoritmos biomédicos y balanceo (Sección 2.2):**
     * **Clásicos:** SVM (hiperplano de margen máximo), Random Forest (*bagging* de árboles) y XGBoost (*gradient boosting* regularizado).
     * **Deep Learning:** *Transfer Learning* en CNNs (AlexNet, MobileNet, InceptionV3, ResNet) y segmentación con **U-Net**.
     * **Control de desbalance:** Uso de **SMOTE** (sobremuestreo sintético de la clase minoritaria) para mitigar el sesgo hacia pacientes sanas.
* **Guion sugerido:**
  > «En nuestro marco teórico vinculamos la biología cervical con la algorítmica. Clínicamente, el Sistema Bethesda clasifica los hallazgos desde negativos (NILM) hasta lesiones de bajo y alto grado (LSIL, HSIL), categorías que estructuran el entrenamiento supervisado. Computacionalmente, contrastamos clasificadores tabulares clásicos —apoyados con SMOTE para manejar el desbalance de clases— frente a arquitecturas convolucionales entrenadas mediante transferencia de aprendizaje.»

---

## Diapositiva 8: Marco Teórico — Inteligencia Artificial Explicable (XAI)
* **Objetivo:** Explicar la diferencia entre interpretabilidad intrínseca y explicabilidad *post-hoc*, y presentar las herramientas utilizadas.
* **Puntos clave que no debes pasar por alto:**
  1. **Modelos intrínsecos vs. Métodos post-hoc:**
     * **Intrínsecos:** Transparentes por diseño (Regresión Logística mediante pesos/odds ratios; Árboles de Decisión mediante reglas lógicas).
     * **Post-hoc:** Auditan modelos complejos de «caja negra» (ensambles y CNNs) ya entrenados, **sin sacrificar su poder predictivo**.
  2. **Tres herramientas clave:**
     * **SHAP (*SHapley Additive exPlanations*):** Basado en teoría de juegos cooperativos; cuantifica el aporte marginal exacto de cada factor de riesgo.
     * **LIME:** Aproxima localmente una predicción individual mediante un modelo lineal interpretable.
     * **Grad-CAM:** Genera mapas de calor sobre la imagen citológica indicando qué regiones celulares activaron la predicción de la red.
  3. **Requisito clínico:** La explicación debe tener coherencia semiológica (sentido biológico para el médico).
* **Guion sugerido:**
  > «En cuanto a la Inteligencia Artificial Explicable, distinguimos entre modelos intrínsecamente transparentes —como la regresión logística o los árboles de decisión— y métodos *post-hoc*, que nos permiten auditar modelos complejos sin reducir su precisión. Destacamos tres técnicas: SHAP, que asigna la contribución de cada variable clínica mediante teoría de juegos; LIME, para aproximaciones lineales locales; y Grad-CAM, que proyecta mapas térmicos sobre las estructuras celulares que activan la red neuronal.»

---

## Diapositiva 9: Estado del Arte — Evidencia en Tabulares, Imágenes y XAI
* **Objetivo:** Resumir qué ha logrado la literatura reciente en datos tabulares, citología digital y AutoML, mostrando el contraste entre métricas reportadas y validez clínica.
* **Puntos clave que no debes pasar por alto:**
  1. **Datos tabulares y AutoML (Secciones 3.1 y 3.3):**
     * En el dataset UCI (858 pacientes), estudios previos reportan hasta `97,99%` de exactitud con Random Forest (AUC `99,74%`) y `97,01%` con XGBoost.
     * Sin embargo, **meta-análisis recientes** (Ramos-Casallas et al., Kondo et al.) advierten falta de validación externa, baja diversidad poblacional y posible contaminación de datos.
     * Auge de enfoques híbridos **AutoML + XAI** (como FSAE y H2O AutoML combinados con SHAP/LIME).
  2. **Imágenes citológicas y XAI (Secciones 3.2 y 3.4):**
     * Evolución desde datasets de **células aisladas** (Herlev, SIPaKMeD) hacia **frotis multicelulares completos** (RIVA, BiCervi, WSI).
     * Altos desempeños reportados en la literatura: MobileNet (AUC `0,97`), InceptionV3 (AUC `0,947`) y segmentación con U-Net (Dice `> 0,85`).
     * **Gran vacío:** Aunque se incluye Grad-CAM visualmente, **menos del 5%** de los trabajos validan esas explicaciones con especialistas clínicos.
* **Guion sugerido:**
  > «Al revisar el estado del arte, observamos cifras muy altas en la literatura: hasta 98% de exactitud en el dataset tabular UCI y AUCs superiores a 0,94 en citología, junto con una evolución desde células recortadas hacia frotis completos como RIVA. No obstante, los meta-análisis recientes advierten problemas de validación externa y sesgos metodológicos, y confirman que menos del 5% de los estudios someten sus mapas de explicabilidad a validación médica formal.»

---

## Diapositiva 10: Brechas Identificadas en el Estado del Arte y Aporte del Estudio
* **Objetivo:** Justificar por qué nuestro experimento es necesario a partir de dos grandes vacíos de la literatura.
* **Puntos clave que no debes pasar por alto:**
  1. **Brecha 1 — Falta de validación cuantitativa y clínica de XAI:** La literatura se limita a mostrar mapas visuales aislados sin medir cuantitativamente su fidelidad ni su utilidad real para el patólogo.
  2. **Brecha 2 — Ausencia de comparación experimental unificada:** Cada artículo usa particiones, preprocesamientos y métricas distintas (a menudo incluyendo variables que causan *data leakage*), impidiendo comparar modelos en igualdad de condiciones.
  3. **Los 3 aportes directos de nuestro trabajo:**
     * **Control estricto de sesgos:** Eliminación de *data leakage* y aplicación de SMOTE exclusivamente en el conjunto de entrenamiento.
     * **Contraste arquitectónico cruzado:** Evaluar las mismas 4 familias de CNN tanto en células aisladas (Herlev, SIPaKMeD) como en frotis multicelulares Bethesda (RIVA).
     * **Doble auditoría XAI:** Integración conjunta de **SHAP** (epidemiológico/tabular) y **Grad-CAM** (morfológico/visual).
* **Guion sugerido:**
  > «Esto nos permite definir dos brechas centrales: primero, la carencia de validación rigurosa en XAI; y segundo, la fragmentación experimental entre estudios que impide comparaciones justas y oculta sesgos como la "Paradoja del Accuracy". Nuestro estudio responde a estas brechas mediante tres aportes: un pipeline libre de fuga de datos, una comparación cruzada de cuatro arquitecturas CNN en tres repositorios citológicos de distinta complejidad, y una doble auditoría post-hoc con SHAP y Grad-CAM.»

---

## Diapositiva 11: Metodología — Conjuntos de Datos Públicos (Tabla 1)
* **Objetivo:** Describir las características técnicas y clínicas de los cuatro datasets utilizados en los experimentos.
* **Puntos clave que no debes pasar por alto:**
  1. **UCI Cervical Cancer Risk Factors (Tabular):** `858` registros y `36` atributos demográficos y conductuales (Hospital Universitario de Caracas), clasificación binaria (Biopsia: CaCu / No CaCu) con fuerte desbalance.
  2. **Herlev (Imágenes unicelulares):** `917` imágenes de células aisladas en `7` categorías de lesiones preneoplásicas; volumen muestral pequeño por clase.
  3. **SIPaKMeD (Imágenes unicelulares):** `966` parches con `4.049` células aisladas clasificadas en `5` categorías morfológicas bien definidas.
  4. **RIVA (Imágenes de frotis multicelular):** `959` campos de visión de `1024×1024 px` con `26.158` anotaciones realizadas por hasta 4 expertos, divididas en las `8` categorías del Sistema Bethesda.
  5. **Proyección futura:** Uso de repositorios como **BiCervi** y láminas completas (**WSI**) para evaluar generalización externa frente al *domain shift*.
* **Guion sugerido:**
  > «En la Tabla 1 detallamos los cuatro repositorios públicos empleados. En la modalidad tabular utilizamos el dataset UCI con 858 pacientes y 36 atributos clínicos. En visión por computadora abarcamos tres niveles de complejidad: Herlev con 917 células aisladas en 7 clases; SIPaKMeD con 4.049 células en 5 categorías morfológicas; y RIVA, un repositorio reciente en *Nature Scientific Data* con más de 26.000 anotaciones en frotis convencionales bajo las 8 categorías del Sistema Bethesda.»

---

## Diapositiva 12: Metodología — Algoritmos y Protocolo Experimental
* **Objetivo:** Explicar con rigor el diseño experimental para garantizar reproducibilidad y ausencia de contaminación entre entrenamiento y prueba.
* **Puntos clave que no debes pasar por alto:**
  1. **Cinco clasificadores tabulares:** Regresión Logística (`max_iter=1000`), SVM (`kernel='rbf'`), Decision Tree, Random Forest (`n_estimators=100`) y XGBoost (`eval_metric='logloss'`).
  2. **Cuatro arquitecturas CNN (Transfer Learning):** AlexNet, MobileNet, InceptionV3 y ResNet.
  3. **Prevención de Data Leakage (Punto metodológico crucial):** En el dataset UCI se **excluyeron** las variables diagnósticas intermedias (*Hinselmann, Schiller y Citology*) para predecir la **Biopsia** únicamente a partir de factores de riesgo clínicos reales (muchos estudios de la literatura inflan su exactitud incluyendo estas pruebas).
  4. **Preprocesamiento limpio:** Imputación por mediana (`SimpleImputer`), estandarización (`StandardScaler`) y balanceo **SMOTE aplicado únicamente al conjunto de entrenamiento**, manteniendo intacta la distribución real en el conjunto de prueba (`15%` hold-out estratificado, `random_state=42`).
  5. **Seis métricas:** Accuracy, Sensibilidad (Recall), Especificidad, Precisión, F1-Score y AUC-ROC.
* **Guion sugerido:**
  > «Nuestro protocolo experimental evalúa cinco clasificadores tabulares y cuatro arquitecturas CNN. Un aspecto metodológico fundamental en datos tabulares fue prevenir la fuga de datos (*data leakage*): excluimos exámenes diagnósticos intermedios como Hinselmann, Schiller y Citología para evaluar la predicción genuina desde factores de riesgo hacia la biopsia. Además, aplicamos imputación por mediana, escalado estándar y sobremuestreo SMOTE exclusivamente dentro del conjunto de entrenamiento, evaluando sobre un conjunto de prueba estratificado con la prevalencia real de la enfermedad.»

---

## Diapositiva 13: Resultados Empíricos en Datos Tabulares (Tabla 2a)
* **Objetivo:** Presentar los resultados cuantitativos en el dataset UCI y demostrar empíricamente la **«Paradoja del Accuracy»**.
* **Puntos clave que no debes pasar por alto:**
  1. **Contraste de métricas en XGBoost y Random Forest:**
     * **XGBoost** alcanza la mayor exactitud (`93.80%`), altísima especificidad (`99.17%`) y la mejor precisión (`50.00%`), pero su **sensibilidad es de solo `12.50%`**.
     * **Random Forest** logra `92.25%` de exactitud y el **mayor AUC-ROC (`76.45%`)**, pero también `12.50%` de sensibilidad.
     * **Decision Tree** alcanza `25.00%` de sensibilidad pero menor exactitud (`87.60%`) y AUC (`58.06%`).
     * **SVM** obtiene `89.15%` de exactitud con **`0.00%` de sensibilidad** (predice casi todo como negativo).
  2. **Significado clínico de la «Paradoja del Accuracy»:** En un conjunto de prueba donde más del 93% de las pacientes son sanas, un modelo puede lograr `~94%` de exactitud acertando a las pacientes sanas mientras **omite al 87,5% de las pacientes con cáncer** (falsos negativos críticos en oncología).
* **Guion sugerido:**
  > «La Tabla 2a revela uno de los hallazgos centrales del estudio: la "Paradoja del Accuracy". Si nos guiamos solo por la exactitud global, XGBoost parece excelente con un 93,80% y una especificidad de 99,17%, o incluso SVM con 89,15%. Pero al examinar la sensibilidad oncológica, vemos que XGBoost y Random Forest solo detectan el 12,50% de los casos positivos bajo el umbral estándar, y SVM el 0%. Es decir, la alta exactitud es una ilusión estadística provocada por el predominio de casos negativos en la población real.»

---

## Diapositiva 14: Capacidad Discriminativa en Datos Tabulares (Curvas ROC — Figura 3)
* **Objetivo:** Explicar por qué los modelos de ensamble sí aprendieron patrones clínicos valiosos a pesar de su baja sensibilidad inicial, y cómo corregirlo mediante *thresholding*.
* **Puntos clave que no debes pasar por alto:**
  1. **Interpretación de la Figura 3 (Curvas ROC):** Aunque la sensibilidad en $p = 0.50$ es baja, **Random Forest (`AUC = 76.45%`)** y **XGBoost (`AUC = 75.41%`)** se despegan claramente de la diagonal de azar (`50%`) y superan ampliamente a Regresión Logística (`50.93%`) y Decision Tree (`58.06%`).
  2. **Diagnóstico del problema:** El modelo sí ordena correctamente el riesgo de las pacientes, pero el **umbral de decisión por defecto ($p = 0.50$)** es demasiado conservador frente a una prevalencia desbalanceada (~15:1).
  3. **Solución clínica (Calibración de umbral / *Threshold moving*):** Reducir el umbral de probabilidad (por ejemplo, hacia el rango $p^* \approx 0.15 - 0.20$) permite desplazarse por la curva ROC para **capturar la gran mayoría de los casos positivos (alta sensibilidad)**, asumiendo una tasa controlada de falsos positivos que pasarán a colposcopía confirmatoria.
* **Guion sugerido:**
  > «¿Significa esto que los modelos tabulares no sirven? La Figura 3 demuestra lo contrario. Las curvas ROC revelan que Random Forest (con 76,45% de AUC) y XGBoost (con 75,41%) sí poseen una capacidad discriminativa latente muy superior al azar. El cuello de botella no es el algoritmo, sino usar el umbral rígido por defecto de 0,50. En un escenario de tamizaje clínico, la solución consiste en calibrar el umbral de probabilidad hacia valores más bajos, priorizando la sensibilidad para que ninguna paciente de alto riesgo quede sin derivación.»

---

## Diapositiva 15: Resultados Empíricos en Imágenes Citológicas (Tabla 2b)
* **Objetivo:** Exponer la comparativa completa de las 4 arquitecturas CNN a través de los 3 datasets citológicos (Herlev, SIPaKMeD y RIVA).
* **Puntos clave que no debes pasar por alto:**
  1. **Desempeño sobresaliente en SIPaKMeD (5 clases):**
     * **AlexNet** lidera de forma contundente con **`93.63%` de exactitud**, **`93.99%` de sensibilidad**, **`98.40%` de especificidad**, `93.76%` de F1-Score y **`99.65%` de AUC-ROC**.
     * Le sigue MobileNet con `49.35%` de exactitud (`81.29%` AUC-ROC).
  2. **Liderazgo en RIVA (8 categorías del Sistema Bethesda):**
     * A pesar de la alta complejidad de 8 clases en frotis completos, **AlexNet** alcanza el primer lugar con **`59.20%` de exactitud**, `50.20%` de sensibilidad, **`93.49%` de especificidad** y **`88.90%` de AUC-ROC**.
     * MobileNet alcanza el segundo puesto con `50.40%` de exactitud (`83.56%` AUC-ROC).
  3. **Resultados en Herlev (7 clases, muestra reducida de 917 imágenes):**
     * **AlexNet** obtiene la mayor exactitud (`45.26%`, AUC `80.39%`), mientras que **MobileNet** logra la mejor sensibilidad (`47.86%`), F1-Score (`47.76%`) y AUC-ROC (`81.66%`).
  4. **Colapso transversal de ResNet:** ResNet registra los peores indicadores en los tres datasets (`16.79%` en Herlev, `33.17%` en SIPaKMeD y `32.89%` en RIVA).
* **Guion sugerido:**
  > «Al pasar a visión por computadora en la Tabla 2b, observamos un patrón muy consistente en las 12 combinaciones evaluadas. AlexNet destaca como la arquitectura líder: en SIPaKMeD alcanza un 93,63% de exactitud y un AUC-ROC de 99,65%; y en el desafiante dataset RIVA de 8 categorías Bethesda logra un 59,20% de exactitud y 88,90% de AUC-ROC, seguido por MobileNet con 50,40%. En Herlev, debido a su reducido tamaño muestral repartido en 7 clases, AlexNet y MobileNet lideran con AUCs superiores al 80%. En contraste, ResNet presenta una caída severa de rendimiento en los tres repositorios.»

---

## Diapositiva 16: Análisis en Visión por Computadora y Recomendaciones
* **Objetivo:** Explicar técnicamente por qué AlexNet superó a arquitecturas más profundas como ResNet50 y formular las recomendaciones de ingeniería de modelos.
* **Puntos clave que no debes pasar por alto:**
  1. **¿Por qué lideró AlexNet (implementada en PyTorch)?** Su menor profundidad relativa y menor cantidad de capas convolucionales evitan el sobreajuste (*overfitting*) en datasets médicos de tamaño moderado y facilitan una convergencia rápida hacia rasgos morfológicos celulares (tamaño nuclear, relación núcleo-citoplasma).
  2. **¿Por qué se estancaron los modelos en Keras y colapsó ResNet50?**
     * En MobileNet e InceptionV3, entrenar únicamente el cabezal denso manteniendo congelada la base preentrenada en ImageNet impuso un techo cercano al `50%` de exactitud.
     * En **ResNet50**, congelar sus bloques profundos impidió adaptar los filtros de alto nivel a texturas citológicas, provocando que el *Early Stopping* detuviera el entrenamiento en la **época 6** por estancamiento de la validación.
  3. **Dos recomendaciones técnicas clave:**
     * **En datos tabulares:** Calibrar el umbral de decisión (*threshold tuning*) sobre Random Forest y XGBoost.
     * **En imágenes citológicas:** Aplicar **Fine-Tuning progresivo** (descongelar gradualmente los bloques convolucionales superiores con tasas de aprendizaje bajas) para adaptar redes profundas al dominio citológico.
* **Guion sugerido:**
  > «¿Cómo explicamos técnicamente esta diferencia entre arquitecturas? Primero, AlexNet en PyTorch se beneficia de una estructura menos profunda que converge rápidamente sobre rasgos morfológicos sin sobreajustarse. Segundo, descubrimos que congelar todas las capas base en arquitecturas muy profundas como ResNet50 bloquea el aprendizaje de texturas celulares finas, activando el *Early Stopping* apenas en la época 6. Por tanto, nuestra recomendación técnica es doble: calibrar umbrales de probabilidad en modelos tabulares y aplicar *fine-tuning* progresivo por bloques en redes convolucionales.»

---

## Diapositiva 17: Evaluación de Explicabilidad de los Algoritmos (Tabla 3)
* **Objetivo:** Presentar el marco de evaluación cualitativa de XAI en tres dimensiones y comparar los 7 algoritmos estudiados.
* **Puntos clave que no debes pasar por alto:**
  1. **Definición de las 3 dimensiones superiores:**
     * **Transparencia:** Claridad de la estructura matemática e inspección directa de parámetros.
     * **Interpretabilidad:** Capacidad de explicar cada predicción a nivel global o local en términos comprensibles.
     * **Auditabilidad:** Facilidad para que un tercero verifique sesgos, errores y trazabilidad mediante herramientas *post-hoc*.
  2. **Lectura de la Tabla 3 (El *trade-off* precisión vs. explicabilidad):**
     * **Regresión Logística y Decision Tree:** Explicabilidad global **Alta** (por coeficientes o reglas), pero menor poder discriminativo en problemas complejos.
     * **XGBoost:** Logra el mejor equilibrio en datos tabulares (**Media-Alta** en Auditabilidad y Explicabilidad Global) gracias a su integración exacta con **TreeSHAP**.
     * **CNN (AlexNet, MobileNet, etc.):** Transparencia intrínseca **Baja** (millones de parámetros), pero alcanzan **Auditabilidad Media** cuando se acoplan con **Grad-CAM y LIME**.
* **Guion sugerido:**
  > «Pasando a nuestra evaluación de explicabilidad en la Tabla 3, calificamos los algoritmos bajo tres dimensiones: transparencia estructural, interpretabilidad diagnóstica y auditabilidad por terceros. Mientras la Regresión Logística y los Árboles de Decisión ofrecen alta transparencia nativa pero menor capacidad predictiva, demostramos que XGBoost alcanza una auditabilidad Media-Alta al integrarse con TreeSHAP. Del mismo modo, aunque las redes convolucionales tienen baja transparencia intrínseca, su auditabilidad se eleva a nivel medio mediante mapas de activación Grad-CAM.»

---

## Diapositiva 18: Auditabilidad en Datos Tabulares — SHAP sobre XGBoost (Figura 1)
* **Objetivo:** Demostrar cómo la auditoría *post-hoc* con TreeSHAP abre la «caja negra» de XGBoost y valida su coherencia epidemiológica.
* **Puntos clave que no debes pasar por alto:**
  1. **Concentración del riesgo en el Top-5 (`75.62%` del impacto total en log-odds):**
     * 1.º **Número de embarazos previos** (`18.06%`)
     * 2.º **Número de parejas sexuales** (`17.00%`)
     * 3.º **Edad de inicio de relaciones sexuales** (`15.67%`)
     * 4.º **Años de uso de anticonceptivos hormonales** (`12.98%`)
     * 5.º **Edad de la paciente** (`11.91%`)
  2. **Explicación de la Figura 1 (Paneles A y B):**
     * **Panel izquierdo (A):** Importancia global media (`mean(|SHAP|)`) que jerarquiza el peso de cada variable clínica.
     * **Panel derecho (B — *Beeswarm plot*):** Muestra la direccionalidad clínica en cada paciente individual (los colores cálidos/rojos indican valores altos de la variable que empujan el riesgo hacia la derecha, es decir, mayor probabilidad de biopsia positiva).
  3. **Coherencia médica y criterio de uso:** Este ranking coincide plenamente con la literatura ginecológica sobre exposición acumulada al VPH, confirmando que XGBoost + SHAP es ideal para **tamizaje poblacional auditable**.
* **Guion sugerido:**
  > «En la Figura 1 auditamos el modelo XGBoost mediante TreeSHAP. Descubrimos que solo cinco variables concentran el 75,62% de la atribución predictiva: número de embarazos previos (18,06%), número de parejas sexuales (17,00%), edad del primer coito (15,67%), años de anticonceptivos hormonales (12,98%) y edad (11,91%). El gráfico *beeswarm* a la derecha permite ver cómo valores altos en estos factores incrementan positivamente los *log-odds* de malignidad. Esto demuestra que el modelo aprende relaciones coherentes con la historia natural de exposición al VPH.»

---

## Diapositiva 19: Auditabilidad Visual en Citología — Mapas de Activación (Figura 2)
* **Objetivo:** Demostrar visualmente que las predicciones de la red convolucional en SIPaKMeD se basan en morfología celular real y no en artefactos de fondo.
* **Puntos clave que no debes pasar por alto:**
  1. **Qué muestra la Figura 2 (5 clases de SIPaKMeD):** Recorre las cinco categorías citológicas (desde células normales superficiales/intermedias hasta células displásicas, koilocitos y metaplásicas), comparando la imagen original, el mapa de calor Grad-CAM y la superposición.
  2. **Verificación morfológica precisa:** Las zonas de máxima activación térmica (rojo/amarillo) se concentran exactamente sobre **núcleos hipercromáticos agrandados, irregularidades de la membrana nuclear y bordes citoplasmáticos alterados** (como los halos perinucleares en koilocitos).
  3. **Descarte de sesgos (*Clever Hans effect*):** El mapa confirma que la red **ignora artefactos de fondo, suciedad del portaobjetos o sesgos de tinción**, entregando al citopatólogo evidencia visual auditable que respalda el `93.63%` de exactitud.
* **Guion sugerido:**
  > «De forma análoga, en la Figura 2 aplicamos Grad-CAM sobre las cinco clases celulares de SIPaKMeD para auditar visualmente a la red convolucional. Como se aprecia en las superposiciones térmicas, la atención de la red se focaliza con precisión sobre los núcleos hipercromáticos y las alteraciones del borde citoplasmático, descartando que el 93,63% de exactitud provenga de artefactos de fondo o sesgos de tinción. Así transformamos una probabilidad numérica en una segunda opinión visualmente justificable ante el citopatólogo.»

---

## Diapositiva 20: Conclusiones, Trabajo Futuro y Agradecimientos
* **Objetivo:** Cerrar la exposición sintetizando las tres conclusiones principales, las tres líneas de investigación futura y el agradecimiento institucional.
* **Puntos clave que no debes pasar por alto:**
  1. **Tres conclusiones principales:**
     * **Paradoja tabular resuelta:** En datos desbalanceados (UCI), una alta exactitud (`93.80%` en XGBoost) oculta una baja sensibilidad (`12.50%`), por lo que es obligatorio calibrar el umbral clínico aprovechando su buen AUC-ROC (`75%–76%`).
     * **Superioridad de CNN adaptadas al dominio:** AlexNet lideró en citología con AUC-ROC de `99.65%` en SIPaKMeD y `88.90%` en RIVA, superando a redes profundas con capas congeladas como ResNet50.
     * **Imperativo de XAI:** Con menos del `5%` de modelos validados clínicamente en la literatura, acoplar **SHAP** y **Grad-CAM** es un requisito innegociable para la adopción médica.
  2. **Tres líneas de trabajo futuro:**
     * Desarrollar **métricas cuantitativas de XAI** (fidelidad, estabilidad) y validarlas con especialistas.
     * Implementar esquemas **Human-in-the-Loop** donde el médico retroalimente y corrija al modelo.
     * Desplegar **Aprendizaje Federado** entre hospitales para entrenar modelos robustos al *domain shift* preservando la privacidad de las pacientes.
  3. **Agradecimientos:** Reconocimiento al proyecto **BIP 40074839-0** del Gobierno Regional de Antofagasta y la Universidad Católica del Norte.
* **Guion sugerido:**
  > «Para concluir, destacamos tres mensajes finales: primero, en el tamizaje clínico tabular no basta con maximizar el *accuracy*; es imperativo calibrar el umbral de decisión para proteger la sensibilidad oncológica. Segundo, en citología cervical, arquitecturas como AlexNet y esquemas de *fine-tuning* progresivo ofrecen una discriminación sobresaliente tanto en células aisladas como en frotis Bethesda. Y tercero, combinar ese rendimiento con auditoría mediante SHAP y Grad-CAM es la llave para superar el efecto "caja negra". Como trabajo futuro proyectamos métricas cuantitativas de XAI, validación *Human-in-the-Loop* con citopatólogos y entrenamiento distribuido mediante Aprendizaje Federado. Muchas gracias por su atención, y quedo atento a sus preguntas.»
