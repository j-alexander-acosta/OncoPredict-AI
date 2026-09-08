"""
Evaluación Comparativa de Modelos de Machine Learning para Detección de Cáncer Cervicouterino
Dataset: Cervical cancer (Risk Factors) - UCI Machine Learning Repository
Variable objetivo: Biopsy
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def load_and_preprocess(filepath: str):
    """
    Carga el dataset, reemplaza '?' por NaN, elimina variables con fuga de información
    y separa predictores (X) y variable objetivo (y).
    """
    df = pd.read_csv(filepath)
    
    # 1. Reemplazar '?' por nulos
    df = df.replace('?', np.nan)
    
    # 2. Excluir diagnósticos intermedios para prevenir data leakage
    leakage_cols = ['Hinselmann', 'Schiller', 'Citology']
    df = df.drop(columns=leakage_cols, errors='ignore')
    
    # 3. Separar X e y
    target_col = 'Biopsy'
    X = df.drop(columns=[target_col]).astype(float)
    y = df[target_col].astype(int)
    
    return X, y


def train_and_evaluate(X, y, random_state: int = 42):
    """
    Realiza la partición estratificada, imputación, estandarización,
    balanceo de clases con SMOTE (solo en train) y evaluación en test.
    """
    # 1. División estratificada (85% train/val, 15% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=random_state
    )
    
    # 2. Imputación con mediana (ajustada en train, aplicada a test)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # 3. Estandarización (ajustada en train, aplicada a test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # 4. Balanceo de clases con SMOTE (estrictamente en conjunto de entrenamiento)
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"--- Distribución de Clases ---")
    print(f"Entrenamiento original: {np.bincount(y_train)} (0: {np.bincount(y_train)[0]}, 1: {np.bincount(y_train)[1]})")
    print(f"Entrenamiento tras SMOTE: {np.bincount(y_train_res)} (0: {np.bincount(y_train_res)[0]}, 1: {np.bincount(y_train_res)[1]})")
    print(f"Prueba (15%): {np.bincount(y_test)} (0: {np.bincount(y_test)[0]}, 1: {np.bincount(y_test)[1]})\n")
    
    # 5. Definición de modelos a evaluar
    models = {
        'Regresión Logística': LogisticRegression(
            random_state=random_state,
            max_iter=1000
        ),
        'SVM': SVC(
            probability=True,
            kernel='rbf',
            random_state=random_state
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ),
        'XGBoost': XGBClassifier(
            eval_metric='logloss',
            random_state=random_state
        )
    }
    
    # 6. Entrenamiento y cómputo de métricas en test
    results = []
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Matriz de confusión para Especificidad y Sensibilidad
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        sensitivity = recall_score(y_test, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = precision_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Modelo': name,
            'Accuracy': f"{accuracy*100:.2f}% ({accuracy:.4f})",
            'Sensibilidad (Recall)': f"{sensitivity*100:.2f}% ({sensitivity:.4f})",
            'Especificidad': f"{specificity*100:.2f}% ({specificity:.4f})",
            'Precisión': f"{precision*100:.2f}% ({precision:.4f})",
            'F1-Score': f"{f1*100:.2f}% ({f1:.4f})",
            'AUC-ROC': f"{auc_roc*100:.2f}% ({auc_roc:.4f})"
        })
        
    results_df = pd.DataFrame(results)
    return results_df


if __name__ == '__main__':
    dataset_path = 'risk_factors_cervical_cancer.csv'
    X, y = load_and_preprocess(dataset_path)
    df_metrics = train_and_evaluate(X, y)
    
    print("================================================================================")
    print(" RESULTADOS DE EVALUACIÓN EN CONJUNTO DE PRUEBA (15% ESTRATIFICADO)")
    print("================================================================================")
    print(df_metrics.to_string(index=False))
