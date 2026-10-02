import pandas as pd
import os
from datetime import datetime

class ExplicabilityReportGenerator:
    """
    Módulo 4: Generador de Reportes de Explicabilidad (XAI)
    Permite consolidar las métricas de interpretabilidad local (Deletion AUC) 
    y explicabilidad global (TCAV) en tablas estructuradas (CSV y Markdown) 
    listas para ser incluidas en la tesis doctoral.
    """
    
    def __init__(self, output_dir="Reportes_XAI"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics_data = []

    def add_record(self, model_name, dataset, class_target, deletion_auc, tcav_score, svm_accuracy):
        """Agrega un registro de evaluación al reporte."""
        self.metrics_data.append({
            "Modelo": model_name,
            "Dataset": dataset,
            "Clase Objetivo": class_target,
            "Deletion AUC ↓ (Local)": round(deletion_auc, 4),
            "TCAV Score ↑ (Global)": f"{round(tcav_score * 100, 2)}%",
            "Exactitud SVM CAV": f"{round(svm_accuracy * 100, 2)}%"
        })

    def generate_reports(self, filename_prefix="XAI_Metrics"):
        """Genera y guarda los reportes en formatos CSV y Markdown."""
        if not self.metrics_data:
            print("No hay datos para generar el reporte.")
            return

        df = pd.DataFrame(self.metrics_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.csv")
        md_path = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.md")
        
        # 1. Exportar a CSV (Para anexos o análisis en Excel/R)
        df.to_csv(csv_path, index=False)
        
        # 2. Exportar a Markdown (Para copiar y pegar directo en la Tesis)
        md_content = "# Tabla Comparativa de Explicabilidad y Confianza (XAI)\n\n"
        md_content += "Esta tabla consolida las métricas cuantitativas que validan la robustez interpretativa de los modelos desarrollados en OncoPredict AI.\n\n"
        md_content += df.to_markdown(index=False)
        md_content += "\n\n**Notas Metodológicas:**\n"
        md_content += "- **Deletion AUC (Interpretabilidad Local):** Mide la degradación de la confianza del modelo al ocultar progresivamente las regiones más importantes según *Score-CAM*. Valores más cercanos a 0 indican mapas de calor más precisos.\n"
        md_content += "- **TCAV Score (Explicabilidad Global):** Mide la sensibilidad direccional del modelo hacia el concepto humano de 'Célula Anormal'. Valores sobre 50% indican una contribución positiva a la clase objetivo.\n"
        md_content += "- **Exactitud SVM CAV:** Indica qué tan bien la capa convolucional separa el concepto estudiado frente al ruido aleatorio.\n"
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        print(f"Reporte CSV generado exitosamente en: {csv_path}")
        print(f"Reporte Markdown generado exitosamente en: {md_path}")
        return csv_path, md_path

if __name__ == "__main__":
    # Prueba del generador de reportes con los datos que obtuvimos en las pruebas
    print("Generando reporte de prueba...")
    report = ExplicabilityReportGenerator()
    
    # Agregar los datos obtenidos en nuestras pruebas técnicas (Fase 2 y Fase 3)
    report.add_record(
        model_name="EfficientNet-B0",
        dataset="SIPaKMeD",
        class_target="Clase 1 (Anormal)",
        deletion_auc=0.2890,
        tcav_score=0.2667,
        svm_accuracy=1.00
    )
    
    # Generar un registro hipotético para ilustrar la tabla comparativa
    report.add_record(
        model_name="MobileNet-V3 (Hipotético)",
        dataset="SIPaKMeD",
        class_target="Clase 1 (Anormal)",
        deletion_auc=0.4500,
        tcav_score=0.1500,
        svm_accuracy=0.85
    )
    
    report.generate_reports()
