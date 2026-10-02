import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from xai_modules.tcav_core import TCAVExtractor, calculate_concept_sensitivity

# Definir un Dataset simple para cargar imágenes de un directorio
class ConceptDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0

def get_gradients(model, input_tensor, target_class):
    """Calcula los gradientes de la predicción respecto a la capa objetivo (bottleneck)."""
    model.eval()
    
    # Necesitamos capturar las activaciones y registrar un hook para sus gradientes
    activations = []
    grads = []
    
    def forward_hook(module, input, output):
        activations.append(output)
        
    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])
        
    # Asumimos que la capa bottleneck es la misma que usamos para XAI local
    target_layer = model.features[-1]
    
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    logits = model(input_tensor)
    score = logits[0, target_class]
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    handle_fw.remove()
    handle_bw.remove()
    
    # Promedio global de los gradientes (para igualar la dimensión del CAV)
    import torch.nn.functional as F
    pooled_grads = F.adaptive_avg_pool2d(grads[0], (1, 1)).squeeze()
    
    return pooled_grads.cpu().numpy()

def compute_tcav_score(model, dataloader_target, cav_vector, target_class, device):
    """
    Calcula el TCAV Score: 
    Proporción de imágenes de la clase objetivo cuyos gradientes tienen una similitud
    direccional positiva con el vector conceptual (CAV).
    """
    positive_count = 0
    total_count = 0
    
    import numpy as np
    
    for images, _ in dataloader_target:
        images = images.to(device)
        for i in range(images.size(0)):
            img = images[i:i+1]
            grad_vector = get_gradients(model, img, target_class)
            
            # Producto punto entre el gradiente direccional y el CAV
            dot_product = np.dot(grad_vector, cav_vector)
            
            if dot_product > 0:
                positive_count += 1
            total_count += 1
            
    if total_count == 0:
        return 0.0
    return positive_count / total_count

def main():
    print("=== FASE 3: Validación TCAV (Explicabilidad Global) ===")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    model_path = os.path.join("Media", "models", "efficientnet_b0_sipakmed.pth")
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        return

    # 1. Cargar el Modelo
    model = models.efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 5) # 5 clases para SIPaKMeD
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. Configurar el Extractor TCAV
    # Usamos la última capa convolucional antes del clasificador
    bottleneck_layer = model.features[-1]
    extractor = TCAVExtractor(model, bottleneck_layer)
    
    # 3. Preparar los datos conceptuales
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    concept_dir = "Media/concepts/concept_anormal"
    random_dir = "Media/concepts/random"
    
    # Verificación de carpetas
    if not os.path.exists(concept_dir) or len(os.listdir(concept_dir)) == 0:
        print(f"\n[!] ACCIÓN REQUERIDA: Debes colocar imágenes en '{concept_dir}' para definir el concepto.")
        return
        
    if not os.path.exists(random_dir) or len(os.listdir(random_dir)) == 0:
        print(f"\n[!] ACCIÓN REQUERIDA: Debes colocar imágenes aleatorias (ruido) en '{random_dir}'.")
        return
        
    print("\nExtrayendo activaciones para el Concepto (Anormal)...")
    dataset_concept = ConceptDataset(concept_dir, transform=transform)
    loader_concept = DataLoader(dataset_concept, batch_size=4, shuffle=False)
    acts_concept = extractor.extract(loader_concept, device)
    
    print("Extrayendo activaciones para el Ruido (Aleatorio)...")
    dataset_random = ConceptDataset(random_dir, transform=transform)
    loader_random = DataLoader(dataset_random, batch_size=4, shuffle=False)
    acts_random = extractor.extract(loader_random, device)
    
    # 4. Entrenar el SVM y obtener el CAV
    print("\nCalculando Sensibilidad del Concepto (Entrenando SVM lineal)...")
    cav_vector, svm_accuracy = calculate_concept_sensitivity(acts_concept, acts_random)
    print(f"-> Exactitud del SVM (Separabilidad del Concepto): {svm_accuracy * 100:.2f}%")
    
    if svm_accuracy < 0.80:
        print("[Advertencia] La separabilidad es menor al 80%. El modelo podría no estar codificando este concepto de manera aislada en esta capa.")
    
    # 5. Calcular el Score TCAV para una clase de prueba
    target_class = 1 # Supongamos que 1 = Anormal en nuestro modelo
    print(f"\nCalculando TCAV Score para la clase objetivo (Index: {target_class})...")
    
    tcav_score = compute_tcav_score(model, loader_concept, cav_vector, target_class, device)
    
    print(f"\n=== RESULTADO FINAL ===")
    print(f"TCAV Score (Concepto vs Clase {target_class}): {tcav_score * 100:.2f}%")
    print("Un score mayor a 50% indica que el concepto estudiado (ej. 'núcleos agrandados' o 'células anormales') contribuye positivamente a la predicción de esta clase.")

if __name__ == "__main__":
    main()
