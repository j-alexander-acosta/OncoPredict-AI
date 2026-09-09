import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def initialize_model(num_classes, device, pretrained_path=None):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model.classifier[4] = nn.Linear(4096, 1024)
    model.classifier[6] = nn.Linear(1024, num_classes)
    
    if pretrained_path and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"Cargados pesos de {pretrained_path}")
        
    return model.to(device)

def train_model(model, trainloader, device, epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 50:.3f}")
                running_loss = 0.0
    print('Finished Training')

def evaluate_model(model, testloader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Especificidad = TN / (TN + FP) usando one-vs-rest
    cnf_matrix = confusion_matrix(y_true, y_pred)
    spec_list = []
    for i in range(num_classes):
        fp = cnf_matrix[:, i].sum() - cnf_matrix[i, i]
        fn = cnf_matrix[i, :].sum() - cnf_matrix[i, i]
        tp = cnf_matrix[i, i]
        tn = cnf_matrix.sum() - (fp + fn + tp)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        spec_list.append(spec)
    spec = np.mean(spec_list)
    
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.5
        
    return acc, sens, spec, prec, f1, auc

def append_metrics_to_json(dataset_name, acc, sens, spec, prec, f1, auc):
    json_path = os.path.join(os.path.dirname(__file__), 'image_metrics.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = []
        
    # Eliminar entradas previas de AlexNet para este dataset
    metrics = [m for m in metrics if not (m['model'] == 'AlexNet' and m['dataset'] == dataset_name)]
    
    metrics.append({
        "dataset": dataset_name,
        "model": "AlexNet",
        "accuracy": f"{acc * 100:.2f}%",
        "sensitivity": f"{sens * 100:.2f}%",
        "specificity": f"{spec * 100:.2f}%",
        "precision": f"{prec * 100:.2f}%",
        "f1": f"{f1 * 100:.2f}%",
        "auc_roc": f"{auc * 100:.2f}%"
    })
    
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.join(os.path.dirname(__file__), 'Media')
    transform = get_transforms()
    
    datasets_to_run = ['Herlev', 'SIPaKMeD', 'RIVA']
    
    for ds_name in datasets_to_run:
        print(f"\n--- Procesando {ds_name} ---")
        if ds_name == 'Herlev':
            train_dir = os.path.join(base_dir, 'Herlev Dataset', 'train')
            test_dir = os.path.join(base_dir, 'Herlev Dataset', 'test')
            train_ds = datasets.ImageFolder(train_dir, transform=transform)
            test_ds = datasets.ImageFolder(test_dir, transform=transform)
            num_classes = len(train_ds.classes)
        else:
            ds_dir = os.path.join(base_dir, ds_name)
            full_ds = datasets.ImageFolder(ds_dir, transform=transform)
            num_classes = len(full_ds.classes)
            train_size = int(0.85 * len(full_ds))
            test_size = len(full_ds) - train_size
            train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=torch.Generator().manual_seed(42))
            
        trainloader = DataLoader(train_ds, batch_size=32, shuffle=True)
        testloader = DataLoader(test_ds, batch_size=32, shuffle=False)
        
        pretrained_path = os.path.join(base_dir, 'models', f'alexnet_{ds_name.lower()}.pth')
        
        if ds_name == 'Herlev' and os.path.exists(pretrained_path):
            print("Evaluando modelo existente...")
            model = initialize_model(num_classes, device, pretrained_path)
        else:
            print("Entrenando nuevo modelo...")
            model = initialize_model(num_classes, device)
            train_model(model, trainloader, device, epochs=2)
            torch.save(model.state_dict(), pretrained_path)
            
        print("Calculando métricas...")
        acc, sens, spec, prec, f1, auc = evaluate_model(model, testloader, device, num_classes)
        print(f"Acc: {acc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}, Prec: {prec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        append_metrics_to_json(ds_name, acc, sens, spec, prec, f1, auc)
        print(f"Métricas actualizadas para {ds_name}")

if __name__ == '__main__':
    main()
