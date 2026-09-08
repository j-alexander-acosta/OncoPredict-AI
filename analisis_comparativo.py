import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# Scikit-learn & Tabular
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

# TensorFlow / Keras & Images
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet, InceptionV3, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def calculate_metrics(y_true, y_pred, y_prob=None, is_multiclass=False):
    if is_multiclass:
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Specificity for multiclass (macro approx)
        cm = confusion_matrix(y_true, y_pred)
        specificities = []
        for i in range(len(cm)):
            tp = cm[i,i]
            fp = cm[:,i].sum() - tp
            fn = cm[i,:].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(spec)
        specificity = np.mean(specificities)
        
        auc_roc = 0
        if y_prob is not None:
            try:
                auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            except Exception as e:
                auc_roc = 0
    else:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)
        
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        auc_roc = 0
        if y_prob is not None:
            try:
                auc_roc = roc_auc_score(y_true, y_prob)
            except:
                auc_roc = 0
            
    return {
        'Accuracy': f"{accuracy*100:.2f}%",
        'Sensitivity (Recall)': f"{sensitivity*100:.2f}%",
        'Specificity': f"{specificity*100:.2f}%",
        'Precision': f"{precision*100:.2f}%",
        'F1-Score': f"{f1*100:.2f}%",
        'AUC-ROC': f"{auc_roc*100:.2f}%"
    }

def run_tabular_pipeline(csv_path):
    print("--- Iniciando Fase 1: Pipeline Tabular (UCI) ---")
    df = pd.read_csv(csv_path)
    
    # Preprocesamiento
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    
    # Imputación
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Eliminar redundantes y definir X, y
    cols_to_drop = ['Hinselmann', 'Schiller', 'Citology', 'Biopsy']
    X = df_imputed.drop(columns=[c for c in cols_to_drop if c in df_imputed.columns])
    y = df_imputed['Biopsy']
    
    # División
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balanceo SMOTE (solo train)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = []
    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics['Model'] = name
        metrics['Pipeline'] = 'Tabular (UCI)'
        results.append(metrics)
        
    return results

def build_transfer_learning_model(base_model_class, num_classes=7, input_shape=(224, 224, 3)):
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Congelar capas base
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def run_image_pipeline(dataset_dir, dataset_name, epochs=20):
    print(f"--- Iniciando Fase 2: Pipeline de Imágenes ({dataset_name}) ---")
    
    if not os.path.exists(dataset_dir):
        print(f"Advertencia: Directorio no encontrado {dataset_dir}.")
        return []
        
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    # Check if there is explicit train/test folders
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Usando estructura train/test para {dataset_name}")
        train_gen = datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )
        test_gen = datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
    else:
        print(f"Usando validation_split para {dataset_name}")
        train_gen = datagen.flow_from_directory(
            dataset_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        test_gen = datagen.flow_from_directory(
            dataset_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
    cnn_models = {
        'MobileNet': MobileNet,
        'InceptionV3': InceptionV3,
        'ResNet50': ResNet50
    }
    
    # En caso de que no haya clases encontradas
    if not train_gen.class_indices:
        print(f"Advertencia: No se encontraron clases en {dataset_dir}")
        return []
        
    num_classes = len(train_gen.class_indices)
    
    # Asegurar que el directorio de modelos exista
    models_dir = os.path.join(os.path.dirname(dataset_dir), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    results = []
    for name, model_class in cnn_models.items():
        print(f"Entrenando {name} (Transfer Learning) en {dataset_name}...")
        model = build_transfer_learning_model(model_class, num_classes=num_classes)
        
        # Entrenamiento
        model.fit(train_gen, epochs=epochs, validation_data=test_gen, verbose=1)
        
        # Guardar el modelo en disco
        model_path = os.path.join(models_dir, f"{name.lower()}_{dataset_name.lower()}.keras")
        model.save(model_path)
        print(f"Modelo {name} guardado en {model_path}")
        
        # Predicciones
        test_gen.reset()
        y_prob = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = test_gen.classes
        
        metrics = calculate_metrics(y_true, y_pred, y_prob, is_multiclass=True)
        metrics['Model'] = name
        metrics['Pipeline'] = f'Image ({dataset_name})'
        results.append(metrics)
        
    return results

def main():
    base_dir = '/Users/alexanderacosta/Documents/Proyectos/Análisis Comparativo'
    tabular_path = os.path.join(base_dir, 'Media', 'risk_factors_cervical_cancer.csv')
    
    # Si no está en Media, usar el del root
    if not os.path.exists(tabular_path):
        tabular_path = os.path.join(base_dir, 'risk_factors_cervical_cancer.csv')
        
    tabular_results = run_tabular_pipeline(tabular_path)
    
    herlev_dir = os.path.join(base_dir, 'Media', 'Herlev Dataset')
    sipakmed_dir = os.path.join(base_dir, 'Media', 'SIPaKMeD')
    riva_dir = os.path.join(base_dir, 'Media', 'RIVA')
    
    all_image_results = []
    
    print("\n--- Procesando Herlev ---")
    all_image_results += run_image_pipeline(herlev_dir, "Herlev", epochs=5)
    
    print("\n--- Procesando SIPaKMeD ---")
    all_image_results += run_image_pipeline(sipakmed_dir, "SIPaKMeD", epochs=5)
    
    print("\n--- Procesando RIVA ---")
    all_image_results += run_image_pipeline(riva_dir, "RIVA", epochs=5)
    
    # Consolidar
    all_results = tabular_results + all_image_results
    df_results = pd.DataFrame(all_results)
    
    # Reordenar columnas
    cols = ['Pipeline', 'Model', 'Accuracy', 'Sensitivity (Recall)', 'Specificity', 'Precision', 'F1-Score', 'AUC-ROC']
    df_results = df_results[cols]
    
    print("\n" + "="*80)
    print(" RESULTADOS CONSOLIDADOS DEL ANÁLISIS COMPARATIVO ".center(80))
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    import json
    metrics_out = []
    for r in all_image_results:
        metrics_out.append({
            'dataset': r['Pipeline'].replace('Image (', '').replace(')', ''),
            'model': r['Model'],
            'accuracy': r['Accuracy'],
            'sensitivity': r['Sensitivity (Recall)'],
            'specificity': r['Specificity'],
            'precision': r['Precision'],
            'f1': r['F1-Score'],
            'auc_roc': r['AUC-ROC']
        })
    with open(os.path.join(base_dir, 'image_metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=4)

if __name__ == '__main__':
    main()
