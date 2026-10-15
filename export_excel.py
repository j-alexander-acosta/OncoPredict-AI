import os
import pandas as pd
from model_service import ModelManager

def export_metrics_to_excel(output_path):
    manager = ModelManager()
    tabular_metrics = manager.get_tabular_metrics()
    image_metrics = manager.get_image_metrics()
    
    # Formatear DataFrames
    df_tab = pd.DataFrame(tabular_metrics)
    df_img = pd.DataFrame(image_metrics)
    
    col_rename = {
        'model': 'Algoritmo',
        'dataset': 'Dataset',
        'accuracy_pct': 'Accuracy',
        'sensitivity_pct': 'Sensibilidad (Recall)',
        'specificity_pct': 'Especificidad',
        'precision_pct': 'Precisión',
        'f1_pct': 'F1-Score',
        'auc_roc_pct': 'AUC-ROC',
        'tp': 'Verdaderos Positivos (TP)',
        'fp': 'Falsos Positivos (FP)',
        'tn': 'Verdaderos Negativos (TN)',
        'fn': 'Falsos Negativos (FN)'
    }
    
    # Columnas deseadas para tabulares
    cols_tab = ['model', 'dataset', 'accuracy_pct', 'sensitivity_pct', 'specificity_pct', 'precision_pct', 'f1_pct', 'auc_roc_pct', 'tp', 'fp', 'tn', 'fn']
    df_tab = df_tab[[c for c in cols_tab if c in df_tab.columns]].rename(columns=col_rename)
    
    # Columnas deseadas para imágenes
    cols_img = ['model', 'dataset', 'accuracy_pct', 'sensitivity_pct', 'specificity_pct', 'precision_pct', 'f1_pct', 'auc_roc_pct']
    df_img = df_img[[c for c in cols_img if c in df_img.columns]].rename(columns=col_rename)
    
    # Exportar a Excel con 2 hojas separadas
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_tab.to_excel(writer, sheet_name='Datos_Tabulares_UCI', index=False)
        df_img.to_excel(writer, sheet_name='Datos_Imagenes_CV', index=False)
        
    print(f"Exportado exitosamente en 2 hojas separadas a: {output_path}")

if __name__ == '__main__':
    export_metrics_to_excel('metricas_evaluacion.xlsx')
