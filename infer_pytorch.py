import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

def infer(model_path, image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)
        
        pt_model = models.alexnet(pretrained=False)
        pt_model.classifier[4] = nn.Linear(4096, 1024)
        pt_model.classifier[6] = nn.Linear(1024, 7)
        pt_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        pt_model.eval()
        
        with torch.no_grad():
            outputs = pt_model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            m_idx = torch.argmax(probs).item()
            m_prob = float(probs[m_idx].item())
            
        print(json.dumps({"success": True, "idx": m_idx, "prob": m_prob}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"success": False, "error": "Invalid arguments"}))
        sys.exit(1)
    infer(sys.argv[1], sys.argv[2])
