import os
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

# Importar nuestros nuevos módulos XAI
from xai_modules.score_cam import ScoreCAM
from xai_modules.quantitative_metrics import calculate_deletion_auc

def initialize_efficientnet(num_classes, device, pretrained_path=None):
    """Inicializa la misma arquitectura usada en train_missing.py"""
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    
    if pretrained_path and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"Modelo cargado desde: {pretrained_path}")
    else:
        print("Advertencia: No se encontró el archivo del modelo preentrenado.")
        
    model.to(device)
    model.eval()
    return model

def main():
    # 1. Configuración Inicial
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # IMPORTANTE: Cambia estas rutas a tus archivos locales reales
    MODEL_PATH = "Media/models/efficientnet_b0_sipakmed.pth"
    NUM_CLASSES = 5 # Asumiendo 5 clases para SIPaKMeD (Ajusta si es distinto)
    
    # Usa una imagen real de tu dataset para la prueba
    # Por ejemplo: "Media/SIPaKMeD/im_Dyskeratotic/001.bmp"
    IMAGE_PATH = "/Users/alexanderacosta/Documents/Proyectos/Análisis Comparativo/Imagenes de Prueba/Copia de Prueba 1.jpeg"
    
    if not os.path.exists(IMAGE_PATH):
        print(f"Por favor, edita IMAGE_PATH en este script para que apunte a una imagen real válida.")
        return

    # 2. Cargar el Modelo
    model = initialize_efficientnet(num_classes=NUM_CLASSES, device=device, pretrained_path=MODEL_PATH)
    
    # 3. Preprocesar la Imagen
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # 4. Generar Mapa de Calor (Módulo 1: Score-CAM)
    # Seleccionamos la última capa convolucional de EfficientNet-B0
    target_layer = model.features[-1] 
    
    print("Generando mapa de calor Score-CAM...")
    score_cam = ScoreCAM(model, target_layer)
    saliency_map = score_cam.generate(input_tensor)
    score_cam.remove_hook()
    
    # 5. Evaluar Cuantitativamente (Módulo 2: Deletion AUC)
    print("Calculando Deletion AUC...")
    # Predicción del modelo para saber qué clase evaluar
    with torch.no_grad():
        target_class = model(input_tensor).argmax(dim=1).item()
        
    saliency_map_tensor = torch.tensor(saliency_map).to(device)
    auc_score = calculate_deletion_auc(model, input_tensor, saliency_map_tensor, target_class)
    
    print(f"Predicción del modelo (Clase Index): {target_class}")
    print(f"Deletion AUC obtenido: {auc_score:.4f} (Más cerca a 0 es mejor)")
    
    import numpy as np
    import matplotlib.cm as cm
    
    # Convertir el mapa de calor a RGB usando el mapa de colores 'jet'
    heatmap = np.uint8(255 * cm.jet(saliency_map)[..., :3])
    
    # Ajustar dimensiones y fusionar
    heatmap_img = Image.fromarray(heatmap).resize(img.size, Image.BILINEAR)
    blended = Image.blend(img.convert('RGB'), heatmap_img, alpha=0.5)
    
    output_filename = "resultado_xai_prueba.png"
    blended.save(output_filename)
    print(f"Resultado guardado como: {output_filename}")

if __name__ == "__main__":
    main()
