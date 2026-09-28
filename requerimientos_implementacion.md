# Guía de Requerimientos de Datos para Implementación Clínica

Este documento detalla los requerimientos técnicos y clínicos exactos para la implementación del sistema **OncoPredict AI** en un entorno médico real (como el Hospital Regional de Antofagasta). El sistema cuenta con dos módulos independientes impulsados por Inteligencia Artificial, y cada uno requiere un tipo de dato completamente distinto.

---

## Módulo 1: Análisis de Imagen (Visión por Computadora - Citología)

Este módulo (que utiliza la red neuronal **EfficientNet-B0**) está diseñado **única y exclusivamente para analizar Citología Digital**. 

### ¿Qué imágenes analiza?
Analiza fotografías **microscópicas a nivel celular**. 
El sistema fue entrenado con bases de datos como Herlev, SIPaKMeD y RIVA, que se componen de capturas de células individuales o pequeños grupos celulares provenientes de frotis de Papanicolaou (Pap).

### Lo que se debe solicitar al laboratorio de Anatomía Patológica:
1. **El Tipo de Muestra:** Imágenes digitales provenientes del microscopio de un examen de Papanicolaou (convencional o base líquida).
2. **Características Visuales:** En la imagen se deben apreciar claramente las células, sus núcleos y el citoplasma. **No sirven** imágenes macroscópicas de colposcopías (tejido del cérvix in-vivo).
3. **Formato:** Archivos `.JPG` o `.PNG` a color (RGB).
4. **Etiquetado Experto (Ground Truth):** Es indispensable que el hospital entregue las imágenes con el diagnóstico clínico asociado emitido por un patólogo (ej. Normal, ASC-US, LSIL, HSIL, Carcinoma). Este etiquetado servirá para comprobar empíricamente si la IA está acertando en la población local.

---

## Módulo 2: Inferencia Clínica Individual (Datos Tabulares)

Este módulo probabilístico (que utiliza algoritmos como **Random Forest y XGBoost**) requiere información sociodemográfica y médica, evaluando los **factores de riesgo** de las pacientes.

### Lo que se debe solicitar a los Registros Clínicos Electrónicos:
Se debe solicitar una base de datos tabular (en formato CSV o Excel) que contenga la mayor cantidad posible de las siguientes variables (basadas en el estándar del Dataset UCI):

* **Demográficos y Reproductivos:** Edad, número de embarazos, edad de la primera relación sexual, número de parejas sexuales.
* **Hábitos:** Antecedentes de tabaquismo (estado actual, años fumando, cantidad de cigarrillos).
* **Anticoncepción:** Uso histórico de anticonceptivos hormonales y Dispositivo Intrauterino (DIU), con sus respectivos años de uso.
* **Infecciones de Transmisión Sexual (ITS):** Todo historial de ITS (VIH, Sífilis, Herpes) y de manera **crítica**, si existe tipificación o diagnóstico de **Virus del Papiloma Humano (VPH)** o condilomatosis.
* **Historial Ginecológico:** Diagnósticos previos de cáncer o lesiones intraepiteliales (NIC/CIN).
* **Variable Objetivo (Biopsia):** El resultado confirmatorio de la biopsia de la paciente (Positivo/Negativo). Este dato es obligatorio para validar matemáticamente la asertividad del modelo.

---

## Consideraciones Éticas y Legales (Contexto Chile)

Para que un recinto de salud pública entregue esta información, el equipo de investigación debe garantizar:

1. **Anonimización Estricta:** La base de datos clínica y las imágenes deben ser entregadas **sin RUT, sin nombres y sin ningún identificador personal**, cumpliendo estrictamente con la Ley N° 19.628 sobre Protección de la Vida Privada.
2. **Autorización Ética:** El protocolo para la extracción de esta data restrospectiva debe ser aprobado previamente por el **Comité Ético Científico (CEC)** correspondiente al Servicio de Salud local.
