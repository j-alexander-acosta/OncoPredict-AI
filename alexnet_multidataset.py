import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

def get_transforms():
    """
    Retorna el bloque de preprocesamiento requerido por AlexNet.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def initialize_model(num_classes, device):
    """
    Carga el modelo preentrenado y adapta la capa de clasificación según la cantidad de clases.
    """
    # Cargar modelo preentrenado de PyTorch Hub
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    
    # Adaptación de la capa de clasificación para la cantidad de clases
    model.classifier[4] = nn.Linear(4096, 1024)
    model.classifier[6] = nn.Linear(1024, num_classes)
    
    # Enviar al dispositivo (GPU/CPU)
    model = model.to(device)
    
    return model

def train_model(model, trainloader, device, epochs=10):
    """
    Bucle de entrenamiento del modelo.
    """
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
            if i % 2000 == 1999:  # Imprimir estadísticas cada 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
    print('Finished Training')

def test_model(model, testloader, device, classes):
    """
    Evaluación del modelo y cálculo de precisión total y por clase.
    """
    model.eval()
    correct = 0
    total = 0
    num_classes = len(classes)
    
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            
            # Asegurar iteración correcta incluso si el último batch es más pequeño
            batch_size = labels.size(0)
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
        
    for i in range(num_classes):
        if class_total[i] > 0:
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
                
    avg = 0
    for i in range(num_classes):
        if class_total[i] > 0:
            temp = (100 * class_correct[i] / class_total[i])
            avg = avg + temp
    avg = avg / num_classes
    print('Average accuracy = ', avg)

def process_dataset(dataset_name, trainloader, testloader, classes, device):
    """
    Canalización completa para un dataset específico.
    """
    print(f"\\n--- Procesando dataset: {dataset_name} ---")
    num_classes = len(classes)
    
    print("Inicializando modelo...")
    model = initialize_model(num_classes, device)
    
    print("Iniciando entrenamiento...")
    train_model(model, trainloader, device, epochs=10)
    
    print("Iniciando pruebas y evaluación...")
    test_model(model, testloader, device, classes)
    
    import os
    save_dir = os.path.join(os.path.dirname(__file__), 'Media', 'models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"alexnet_{dataset_name.lower()}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Modelo guardado exitosamente en: {save_path}")
    
    print(f"--- Fin de procesamiento de {dataset_name} ---\n")

def main(datasets):
    """
    Itera sobre la lista de datasets.
    datasets: lista de diccionarios con la estructura:
      {'name': str, 'trainloader': DataLoader, 'testloader': DataLoader, 'classes': list[str]}
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")
    
    for dataset_info in datasets:
        process_dataset(
            dataset_name=dataset_info['name'],
            trainloader=dataset_info['trainloader'],
            testloader=dataset_info['testloader'],
            classes=dataset_info['classes'],
            device=device
        )

# Ejemplo de uso
if __name__ == '__main__':
    import os
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'Media', 'Herlev Dataset')
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Cargando imágenes de {data_dir} ...")
        transform = get_transforms()
        
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
        
        trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        datasets_to_process = [{
            'name': 'Herlev',
            'trainloader': trainloader,
            'testloader': testloader,
            'classes': train_dataset.classes
        }]
        
        main(datasets_to_process)
    else:
        print(f"Error: No se encontró la carpeta train o test en {data_dir}")

