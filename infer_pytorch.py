import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importar módulos XAI
from xai_modules.score_cam import ScoreCAM
from xai_modules.quantitative_metrics import calculate_deletion_auc

def infer(model_path, image_path):
    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        if "efficientnet_b0_colposcopy" in model_path.lower():
            pt_model = models.efficientnet_b0(pretrained=False)
            num_ftrs = pt_model.classifier[1].in_features
            pt_model.classifier[1] = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 3)
            )
        elif "efficientnet_b0" in model_path.lower():
            pt_model = models.efficientnet_b0(pretrained=False)
            num_ftrs = pt_model.classifier[1].in_features
            pt_model.classifier[1] = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 7)
            )
        else:
            pt_model = models.alexnet(pretrained=False)
            pt_model.classifier[4] = nn.Linear(4096, 1024)
            pt_model.classifier[6] = nn.Linear(1024, 7)
            
        pt_model.load_state_dict(torch.load(model_path, map_location=device))
        pt_model.to(device)
        pt_model.eval()
        
        with torch.no_grad():
            outputs = pt_model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            m_idx = torch.argmax(probs).item()
            m_prob = float(probs[m_idx].item())
            
        xai_info = None
        if "efficientnet" in model_path.lower():
            try:
                target_layer = pt_model.features[-1]
                score_cam = ScoreCAM(pt_model, target_layer)
                saliency_map = score_cam.generate(img_tensor)
                score_cam.remove_hook()
                
                saliency_map_tensor = torch.tensor(saliency_map)
                auc_score = calculate_deletion_auc(pt_model, img_tensor, saliency_map_tensor, m_idx)
                
                output_dir = os.path.join(os.path.dirname(__file__), 'static', 'outputs')
                os.makedirs(output_dir, exist_ok=True)
                
                import time
                filename = f"xai_{int(time.time())}.png"
                out_path = os.path.join(output_dir, filename)
                
                import numpy as np
                import matplotlib.cm as cm
                
                # Convertir [0,1] mapa a RGB usando colormap 'jet' de matplotlib
                heatmap = np.uint8(255 * cm.jet(saliency_map)[..., :3])
                
                # Redimensionar al tamaño original y mezclar con PIL
                heatmap_img = Image.fromarray(heatmap).resize(img.size, Image.BILINEAR)
                blended = Image.blend(img.convert('RGB'), heatmap_img, alpha=0.5)
                blended.save(out_path)
                
                xai_info = {
                    "auc": round(float(auc_score), 4),
                    "image_url": f"/static/outputs/{filename}"
                }
            except Exception as e:
                pass
            
        print(json.dumps({"success": True, "idx": m_idx, "prob": m_prob, "xai": xai_info}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"success": False, "error": "Invalid arguments"}))
        sys.exit(1)
    infer(sys.argv[1], sys.argv[2])
