import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import auc

def calculate_deletion_auc(model, input_tensor, saliency_map, target_class, steps=100):
    """
    Calcula el Deletion AUC oscureciendo progresivamente los píxeles de mayor a menor importancia.
    - AUC menor = Mejor mapa de calor (la probabilidad cayó rápidamente).
    """
    device = input_tensor.device
    model.eval()
    
    b, c, h, w = input_tensor.size()
    
    # Aplanar el mapa de calor para ordenar los píxeles
    if isinstance(saliency_map, torch.Tensor):
        flat_saliency = saliency_map.detach().cpu().numpy().flatten()
    else:
        flat_saliency = saliency_map.flatten()
    # Índices ordenados de mayor a menor importancia
    sort_indices = np.argsort(flat_saliency)[::-1]
    
    # Probabilidad base (100% de la imagen visible)
    with torch.no_grad():
        base_logits = model(input_tensor)
        base_prob = F.softmax(base_logits, dim=1)[0, target_class].item()
    
    probabilities = [base_prob]
    fractions = [0.0]
    
    # Crear una copia de la imagen que iremos oscureciendo
    masked_img = input_tensor.clone()
    
    # Número de píxeles a oscurecer por cada paso
    pixels_per_step = len(flat_saliency) // steps
    
    with torch.no_grad():
        for i in range(1, steps + 1):
            # Seleccionar el siguiente lote de píxeles importantes a borrar
            start_idx = (i - 1) * pixels_per_step
            end_idx = min(i * pixels_per_step, len(flat_saliency))
            idx_to_mask = sort_indices[start_idx:end_idx]
            
            # Convertir índices a coordenadas 2D (h, w)
            y_coords = idx_to_mask // w
            x_coords = idx_to_mask % w
            
            # Oscurecer (Set to 0 o valor medio del dataset) en los 3 canales
            masked_img[0, :, y_coords, x_coords] = 0.0 
            
            # Forward pass con la imagen parcialmente oscurecida
            logits = model(masked_img)
            prob = F.softmax(logits, dim=1)[0, target_class].item()
            
            probabilities.append(prob)
            fractions.append(i / steps)
            
    # Calcular el Área Bajo la Curva (AUC)
    deletion_auc = auc(fractions, probabilities)
    return deletion_auc
