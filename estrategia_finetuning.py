import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def configurar_finetuning_progresivo(num_clases=8, epoca_actual=0):
    """
    Estrategia de Fine-Tuning Progresivo para mitigar el estancamiento morfológico.
    Diseñado para su integración directa en el agente de ejecución Antigravity.
    """
    # 1. Instanciación del modelo pre-entrenado base (SOTA alternativo a ResNet)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # 2. Reconfiguración del clasificador ginecológico (Fully Connected Layer)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_clases)
    )
    
    # 3. Protocolo de Descongelamiento Progresivo según la fase del experimento
    if epoca_actual < 5:
        # Fase Inicial: Congelamiento estricto del extractor de características
        for param in model.features.parameters():
            param.requires_grad = False
        print("[Antigravity] Fase 1: Extractor congelado. Optimizando solo el clasificador superior.")
        optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
        
    else:
        # Fase Avanzada: Descongelamiento de los bloques convolucionales superiores
        # Se liberan los bloques profundos (features[6] y features[7]) para adaptación morfológica
        for name, child in model.features.named_children():
            if int(name) >= 6:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False
                    
        print("[Antigravity] Fase 2: Bloques convolucionales superiores desfreezados.")
        # Tasas de aprendizaje discriminativas para proteger los pesos pre-entrenados de bajo nivel
        optimizer = optim.Optimizer([
            {'params': model.features[6].parameters(), 'lr': 1e-5},
            {'params': model.features[7].parameters(), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 1e-4}
        ], lr=1e-4)
        
    # 4. Función de pérdida diseñada para contrarrestar el desbalance de Bethesda
    criterion = nn.CrossEntropyLoss() # Recomendable transicionar a Focal Loss si persiste el sesgo
    
    return model, optimizer, criterion
