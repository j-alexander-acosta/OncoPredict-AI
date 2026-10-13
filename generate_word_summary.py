from docx import Document
import sys

try:
    doc = Document()
    doc.add_heading('Resumen de Configuración e Hiperparámetros', 0)

    # Table 1: Pipeline
    doc.add_heading('Resumen de Configuración y Pipeline (Datos UCI)', level=2)
    pipeline_data = [
        ["Fase del Pipeline", "Configuración / Parámetro Aplicado", "Descripción"],
        ["Prevención de Fuga (Data Leakage)", "drop(columns=['Hinselmann', 'Schiller', 'Citology'])", "Se eliminan variables de diagnóstico intermedio que sesgarían el modelo hacia la variable objetivo (Biopsy)."],
        ["División de Datos (Hold-out)", "test_size=0.15", "Se reserva un 15% del dataset exclusivamente para prueba (evaluación final)."],
        ["Estratificación", "stratify=y", "Garantiza que la proporción original de clases positivas y negativas se mantenga en entrenamiento y prueba."],
        ["Tratamiento de Nulos", "SimpleImputer(strategy='median')", "Los valores faltantes ('?') se reemplazan por la mediana de cada columna."],
        ["Escalado / Estandarización", "StandardScaler()", "Ajuste para que las variables numéricas tengan media 0 y varianza 1."],
        ["Balanceo de Clases", "SMOTE(random_state=42)", "Generación de muestras sintéticas para balancear las clases, aplicado únicamente al conjunto de entrenamiento."],
        ["Reproducibilidad", "random_state=42", "Semilla fija en todas las fases estocásticas para que los resultados sean deterministas y reproducibles."]
    ]

    table1 = doc.add_table(rows=1, cols=3)
    table1.style = 'Table Grid'
    hdr_cells1 = table1.rows[0].cells
    hdr_cells1[0].text = pipeline_data[0][0]
    hdr_cells1[1].text = pipeline_data[0][1]
    hdr_cells1[2].text = pipeline_data[0][2]

    for row in pipeline_data[1:]:
        row_cells = table1.add_row().cells
        row_cells[0].text = row[0]
        row_cells[1].text = row[1]
        row_cells[2].text = row[2]

    # Table 2: Hyperparameters
    doc.add_heading('Resumen de Hiperparámetros por Modelo', level=2)
    hyperparam_data = [
        ["Modelo", "Hiperparámetros Clave", "Notas"],
        ["Regresión Logística", "max_iter=1000, random_state=42", "Se aumentó el límite de iteraciones (1000) para garantizar la convergencia del optimizador."],
        ["SVM (Vectores de Soporte)", "kernel='rbf', probability=True, random_state=42", "probability=True es necesario para poder calcular la métrica AUC-ROC en la evaluación."],
        ["Decision Tree", "random_state=42 (Sin límite de profundidad)", "Árbol de decisión estándar utilizando los parámetros por defecto de scikit-learn."],
        ["Random Forest", "n_estimators=100, random_state=42", "Ensamblado de 100 árboles de decisión para mayor robustez frente al sobreajuste."],
        ["XGBoost", "eval_metric='logloss', random_state=42", "Función de pérdida logarítmica, óptima para el problema de clasificación binaria abordado."]
    ]

    table2 = doc.add_table(rows=1, cols=3)
    table2.style = 'Table Grid'
    hdr_cells2 = table2.rows[0].cells
    hdr_cells2[0].text = hyperparam_data[0][0]
    hdr_cells2[1].text = hyperparam_data[0][1]
    hdr_cells2[2].text = hyperparam_data[0][2]

    for row in hyperparam_data[1:]:
        row_cells = table2.add_row().cells
        row_cells[0].text = row[0]
        row_cells[1].text = row[1]
        row_cells[2].text = row[2]

    output_path = '/Users/alexanderacosta/Documents/Proyectos/Análisis Comparativo/Resumen_Configuracion_e_Hiperparametros.docx'
    doc.save(output_path)
    print(f"Word generado exitosamente en: {output_path}")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
