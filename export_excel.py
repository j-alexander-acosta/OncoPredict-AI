import os
import pandas as pd
from model_service import ModelManager

def export_metrics_to_excel(output_path):
    manager = ModelManager()
    metrics = manager.get_metrics()
    
    # Formatear a DataFrame
    df = pd.DataFrame(metrics)
    
    # Renombrar columnas para que se vean bien
    df.rename(columns={
        'model': 'Modelo',
        'dataset': 'Dataset',
        'type': 'Tipo',
        'accuracy': 'Exactitud (Accuracy)',
        'recall': 'Sensibilidad (Recall)',
        'sensitivity': 'Sensibilidad (Recall)',
        'specificity': 'Especificidad',
        'precision': 'Precisión',
        'f1_score': 'F1-Score',
        'auc_roc': 'AUC-ROC'
    }, inplace=True)
    
    # Convertir valores a porcentaje legible (ej: 0.938 -> 93.80%)
    for col in ['Exactitud (Accuracy)', 'Sensibilidad (Recall)', 'Especificidad', 'Precisión', 'F1-Score', 'AUC-ROC']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{(x * 100):.2f}%" if pd.notnull(x) else "N/A")
        elif col == 'Sensibilidad (Recall)' and 'sensitivity' in df.columns:
            df.rename(columns={'sensitivity': 'Sensibilidad (Recall)'}, inplace=True)
            df['Sensibilidad (Recall)'] = df['Sensibilidad (Recall)'].apply(lambda x: f"{(x * 100):.2f}%" if pd.notnull(x) else "N/A")
    
    # Exportar a Excel
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"Exportado exitosamente a {output_path}")

if __name__ == '__main__':
    # Guardar en la carpeta actual y en la carpeta de artefactos de Gemini
    export_metrics_to_excel('metricas_evaluacion.xlsx')
    export_metrics_to_excel('/Users/alexanderacosta/.gemini/antigravity-ide/brain/df4cd299-5c65-4d16-b744-61f39768610d/metricas_evaluacion.xlsx')
