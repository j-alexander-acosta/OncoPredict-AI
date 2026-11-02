import os
import json
import copy
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'

def evaluate_predictions(y_true, y_pred, y_prob, num_classes):
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

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

def update_metrics_json(dataset_name, model_name, acc, sens, spec, prec, f1, auc):
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image_metrics.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = []

    entry = {
        "dataset": dataset_name,
        "model": model_name,
        "accuracy": f"{acc * 100:.2f}%",
        "sensitivity": f"{sens * 100:.2f}%",
        "specificity": f"{spec * 100:.2f}%",
        "precision": f"{prec * 100:.2f}%",
        "f1": f"{f1 * 100:.2f}%",
        "auc_roc": f"{auc * 100:.2f}%"
    }

    replaced = False
    for idx, m in enumerate(metrics):
        if m.get('dataset') == dataset_name and m.get('model') == model_name:
            metrics[idx] = entry
            replaced = True
            break
    if not replaced:
        metrics.append(entry)

    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[JSON] Actualizado {model_name} en {dataset_name}: Acc={acc*100:.2f}%, Sens={sens*100:.2f}%, Spec={spec*100:.2f}%, Prec={prec*100:.2f}%, F1={f1*100:.2f}%, AUC={auc*100:.2f}%", flush=True)

class InMemoryImageDataset(Dataset):
    def __init__(self, images_uint8, labels):
        self.images = images_uint8
        self.labels = labels
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert HWC uint8 to CHW float32 normalized
        arr = self.images[idx]
        t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
        t = (t - self.mean) / self.std
        return t, int(self.labels[idx])

def load_riva_in_memory(ds_dir):
    t0 = time.time()
    folder_ds = datasets.ImageFolder(ds_dir)
    samples = folder_ds.samples
    num_classes = len(folder_ds.classes)
    print(f"Cargando {len(samples)} imágenes de RIVA ({num_classes} clases: {folder_ds.classes}) en memoria con 8 hilos...", flush=True)

    resize_crop = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    def _load_one(sample):
        path, label = sample
        with Image.open(path) as img:
            img = img.convert('RGB')
            img = resize_crop(img)
            return np.asarray(img, dtype=np.uint8), label

    images = np.empty((len(samples), 224, 224, 3), dtype=np.uint8)
    labels = np.empty(len(samples), dtype=np.int64)

    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (arr, lbl) in enumerate(ex.map(_load_one, samples)):
            images[i] = arr
            labels[i] = lbl

    print(f"Dataset cargado en {time.time() - t0:.1f}s ({images.nbytes / 1024 / 1024:.1f} MB en RAM).", flush=True)
    return images, labels, num_classes, folder_ds.classes

def train_pytorch_model(mem_ds, num_classes, base_dir, use_efficientnet=True, epochs=15, patience=4):
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    train_size = int(0.85 * len(mem_ds))
    test_size = len(mem_ds) - train_size
    train_ds, test_ds = random_split(mem_ds, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(train_ds, batch_size=64, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=64, shuffle=False)

    if use_efficientnet:
        model_name = "EfficientNet_B0"
        save_name = "efficientnet_b0_riva.pth"
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    else:
        model_name = "AlexNet"
        save_name = "alexnet_riva.pth"
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model.classifier[4] = nn.Linear(4096, 1024)
        model.classifier[6] = nn.Linear(1024, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if not use_efficientnet:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print(f"\n=== Entrenando {model_name} en RIVA ({len(mem_ds)} imgs, {num_classes} clases) en {device} ===", flush=True)
    for epoch in range(epochs):
        t0 = time.time()
        if use_efficientnet:
            if epoch < 3:
                for param in model.features.parameters():
                    param.requires_grad = False
                if epoch == 0:
                    print("[Fase 1] Extractor congelado. Optimizando clasificador.", flush=True)
                optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
            else:
                for name, child in model.features.named_children():
                    if int(name) >= 5:
                        for param in child.parameters():
                            param.requires_grad = True
                    else:
                        for param in child.parameters():
                            param.requires_grad = False
                if epoch == 3:
                    print("[Fase 2] Bloques convolucionales superiores descongelados.", flush=True)
                optimizer = optim.Adam([
                    {'params': model.features[5].parameters(), 'lr': 5e-5},
                    {'params': model.features[6].parameters(), 'lr': 5e-5},
                    {'params': model.features[7].parameters(), 'lr': 5e-5},
                    {'params': model.classifier.parameters(), 'lr': 2e-4}
                ], lr=1e-4)

        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        val_loss /= len(testloader)
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} ({dt:.1f}s) - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping activado.", flush=True)
                break

    model.load_state_dict(best_model_wts)
    save_path = os.path.join(base_dir, 'models', save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Modelo guardado en {save_path}", flush=True)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    acc, sens, spec, prec, f1, auc = evaluate_predictions(
        np.array(all_labels), np.array(all_preds), np.array(all_probs), num_classes
    )
    update_metrics_json("RIVA", model_name, acc, sens, spec, prec, f1, auc)

def train_keras_models(images_uint8, labels, num_classes, base_dir):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split

    models_dir = os.path.join(base_dir, 'models')

    # Partición 80/20 estratificada para Keras
    idx_all = np.arange(len(labels))
    train_idx, val_idx = train_test_split(idx_all, test_size=0.20, stratify=labels, random_state=42)

    y_train_cat = tf.keras.utils.to_categorical(labels[train_idx], num_classes=num_classes)
    y_val_cat = tf.keras.utils.to_categorical(labels[val_idx], num_classes=num_classes)
    y_val_true = labels[val_idx]

    def batch_extract(feat_extractor, indices, batch_size=128):
        feats = []
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_x = images_uint8[batch_idx].astype(np.float32) / 255.0
            f = feat_extractor(batch_x, training=False).numpy()
            feats.append(f)
        return np.vstack(feats)

    for name in ['MobileNet', 'InceptionV3', 'ResNet50']:
        print(f"\n=== Entrenando {name} (Keras Transfer Learning) en RIVA ===", flush=True)
        ref_path = os.path.join(models_dir, f"{name.lower()}_herlev.keras")
        ref_model = tf.keras.models.load_model(ref_path)
        base_model = ref_model.layers[0]
        base_model.trainable = False

        feat_extractor = Sequential([base_model, GlobalAveragePooling2D()])
        t0 = time.time()
        print(f"Extrayendo características con {name} congelado...", flush=True)
        X_train_feats = batch_extract(feat_extractor, train_idx)
        X_val_feats = batch_extract(feat_extractor, val_idx)
        print(f"Características extraídas en {time.time()-t0:.1f}s: train={X_train_feats.shape}, val={X_val_feats.shape}", flush=True)

        np.random.seed(42)
        tf.random.set_seed(42)
        head = Sequential([
            Input(shape=(X_train_feats.shape[1],)),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        head.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        head.fit(
            X_train_feats, y_train_cat,
            validation_data=(X_val_feats, y_val_cat),
            epochs=50,
            batch_size=32,
            callbacks=[es],
            shuffle=True,
            verbose=0
        )

        full_model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        full_model.layers[2].set_weights(head.layers[0].get_weights())
        full_model.layers[3].set_weights(head.layers[1].get_weights())
        full_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        save_path = os.path.join(models_dir, f"{name.lower()}_riva.keras")
        full_model.save(save_path)
        print(f"Modelo {name} guardado en {save_path}", flush=True)

        y_prob = head.predict(X_val_feats, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        acc, sens, spec, prec, f1, auc = evaluate_predictions(y_val_true, y_pred, y_prob, num_classes)
        update_metrics_json("RIVA", name, acc, sens, spec, prec, f1, auc)

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Media')
    ds_dir = os.path.join(base_dir, 'RIVA')
    images_uint8, labels, num_classes, classes = load_riva_in_memory(ds_dir)
    mem_ds = InMemoryImageDataset(images_uint8, labels)

    train_pytorch_model(mem_ds, num_classes, base_dir, use_efficientnet=True, epochs=15, patience=4)
    train_pytorch_model(mem_ds, num_classes, base_dir, use_efficientnet=False, epochs=15, patience=4)
    train_keras_models(images_uint8, labels, num_classes, base_dir)
    print("\n[OK] Entrenamiento de los 5 modelos en RIVA completado exitosamente.", flush=True)
