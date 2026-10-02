import torch
import torch.nn.functional as F
import numpy as np

class ScoreCAM:
    """
    Implementación de Score-CAM para generar mapas de activación sin gradientes.
    Ideal para alta fidelidad espacial en imágenes médicas.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        self.activations = None
        
        # Hook para capturar las activaciones en el pase hacia adelante (forward)
        def hook_fn(module, input, output):
            self.activations = output
        self.hook = self.target_layer.register_forward_hook(hook_fn)

    def generate(self, input_tensor, target_class=None):
        """
        Genera el mapa de calor Score-CAM.
        input_tensor: Tensor de imagen (1, C, H, W) preprocesado.
        """
        device = input_tensor.device
        b, c, h, w = input_tensor.size()
        
        # 1. Forward pass inicial para obtener las activaciones y la clase predicha
        with torch.no_grad():
            logits = self.model(input_tensor)
            if target_class is None:
                target_class = logits.argmax(dim=1).item()
        
        activations = self.activations # (1, num_channels, H_map, W_map)
        num_channels = activations.size(1)
        
        # 2. Upsample de las activaciones a la resolución original (224x224)
        # Interpolación bilineal para mantener bordes suaves.
        upsampled_activations = F.interpolate(
            activations, size=(h, w), mode='bilinear', align_corners=False
        )
        
        score_saliency_map = torch.zeros((1, 1, h, w), device=device)
        
        # 3. Fase de Evaluación (Score): Máscara por cada canal
        with torch.no_grad():
            for i in range(num_channels):
                saliency_map = upsampled_activations[:, i:i+1, :, :]
                
                # Normalización Min-Max para escalar la máscara entre [0, 1]
                min_val = saliency_map.min()
                max_val = saliency_map.max()
                if max_val - min_val > 1e-7:
                    norm_saliency_map = (saliency_map - min_val) / (max_val - min_val)
                else:
                    norm_saliency_map = saliency_map
                
                # Proyectar máscara sobre la imagen original (Element-wise multiplication)
                masked_input = input_tensor * norm_saliency_map
                
                # Forward pass de la imagen enmascarada
                masked_logits = self.model(masked_input)
                # Softmax para obtener la probabilidad de la clase objetivo
                score = F.softmax(masked_logits, dim=1)[0, target_class]
                
                # 4. Combinación Lineal: Activación ponderada por el "score" de confianza
                score_saliency_map += score * saliency_map

        # 5. Aplicar ReLU final (solo nos interesan características positivas para la clase)
        score_saliency_map = F.relu(score_saliency_map)
        
        # Normalización final del mapa de calor resultante
        min_val = score_saliency_map.min()
        max_val = score_saliency_map.max()
        score_saliency_map = (score_saliency_map - min_val) / (max_val - min_val + 1e-7)
        
        return score_saliency_map.squeeze().cpu().numpy()

    def remove_hook(self):
        self.hook.remove()
