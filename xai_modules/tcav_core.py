import torch
import torch.nn.functional as F
import numpy as np
from sklearn.svm import SVC

class TCAVExtractor:
    def __init__(self, model, bottleneck_layer):
        self.model = model
        self.bottleneck_layer = bottleneck_layer
        self.model.eval()
        self.activations = []
        
        def hook_fn(module, input, output):
            # Global Average Pooling para vectorizar la activación convolucional
            pooled_output = F.adaptive_avg_pool2d(output, (1, 1)).squeeze()
            self.activations.append(pooled_output.detach().cpu().numpy())
            
        self.hook = self.bottleneck_layer.register_forward_hook(hook_fn)

    def extract(self, dataloader, device):
        """Pasa imágenes por el modelo y recolecta las activaciones latentes."""
        self.activations = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                _ = self.model(images)
        return np.vstack(self.activations)

def calculate_concept_sensitivity(activations_concept, activations_random):
    """
    Entrena un SVM lineal para separar las activaciones del "Concepto" de 
    las activaciones "Aleatorias", derivando el Vector de Activación del Concepto (CAV).
    """
    # Etiquetas: 1 para Concepto, 0 para Aleatorio
    X = np.vstack([activations_concept, activations_random])
    y = np.array([1] * len(activations_concept) + [0] * len(activations_random))
    
    # Entrenar SVM Lineal
    svm = SVC(kernel='linear')
    svm.fit(X, y)
    
    # El vector ortogonal al hiperplano de decisión es nuestro CAV
    cav_vector = svm.coef_[0]
    
    # Precisión del SVM indica si el concepto está realmente codificado en esa capa
    accuracy = svm.score(X, y)
    
    return cav_vector, accuracy
