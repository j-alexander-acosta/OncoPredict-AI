import pandas as pd

# Data for Pipeline
pipeline_data = {
    "Fase del Pipeline": [
        "Prevención de Fuga (Data Leakage)",
        "División de Datos (Hold-out)",
        "Estratificación",
        "Tratamiento de Nulos",
        "Escalado / Estandarización",
        "Balanceo de Clases",
        "Reproducibilidad"
    ],
    "Configuración / Parámetro Aplicado": [
        "drop(columns=['Hinselmann', 'Schiller', 'Citology'])",
        "test_size=0.15",
        "stratify=y",
        "SimpleImputer(strategy='median')",
        "StandardScaler()",
        "SMOTE(random_state=42)",
        "random_state=42"
    ],
    "Descripción": [
        "Se eliminan variables de diagnóstico intermedio que sesgarían el modelo hacia la variable objetivo (Biopsy).",
        "Se reserva un 15% del dataset exclusivamente para prueba (evaluación final).",
        "Garantiza que la proporción original de clases positivas y negativas se mantenga en entrenamiento y prueba.",
        "Los valores faltantes ('?') se reemplazan por la mediana de cada columna.",
        "Ajuste para que las variables numéricas tengan media 0 y varianza 1.",
        "Generación de muestras sintéticas para balancear las clases, aplicado únicamente al conjunto de entrenamiento.",
        "Semilla fija en todas las fases estocásticas para que los resultados sean deterministas y reproducibles."
    ]
}

# Data for Hyperparameters
hyperparam_data = {
    "Modelo": [
        "Regresión Logística",
        "SVM (Vectores de Soporte)",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],
    "Hiperparámetros Clave": [
        "max_iter=1000, random_state=42",
        "kernel='rbf', probability=True, random_state=42",
        "random_state=42 (Sin límite de profundidad)",
        "n_estimators=100, random_state=42",
        "eval_metric='logloss', random_state=42"
    ],
    "Notas": [
        "Se aumentó el límite de iteraciones (1000) para garantizar la convergencia del optimizador.",
        "probability=True es necesario para poder calcular la métrica AUC-ROC en la evaluación.",
        "Árbol de decisión estándar utilizando los parámetros por defecto de scikit-learn.",
        "Ensamblado de 100 árboles de decisión para mayor robustez frente al sobreajuste.",
        "Función de pérdida logarítmica, óptima para el problema de clasificación binaria abordado."
    ]
}

df_pipeline = pd.DataFrame(pipeline_data)
df_hyperparams = pd.DataFrame(hyperparam_data)

# Create an excel writer
output_path = '/Users/alexanderacosta/Documents/Proyectos/Análisis Comparativo/Resumen_Configuracion_e_Hiperparametros.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_pipeline.to_excel(writer, sheet_name='Pipeline_Preprocesamiento', index=False)
    df_hyperparams.to_excel(writer, sheet_name='Hiperparametros_Modelos', index=False)

print(f"Excel generado exitosamente en: {output_path}")
