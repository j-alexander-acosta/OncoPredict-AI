"""
Servicio de Machine Learning para Diagnóstico Predictivo de Cáncer Cervicouterino.
Gestiona el preprocesamiento, entrenamiento de los 5 clasificadores biomédicos,
cómputo de métricas y la inferencia interactiva.
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
import numpy as np
import pandas as pd
from PIL import Image
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

DATASET_FILE = os.path.join(os.path.dirname(__file__), 'Media', 'risk_factors_cervical_cancer.csv')
if not os.path.exists(DATASET_FILE):
    DATASET_FILE = os.path.join(os.path.dirname(__file__), 'risk_factors_cervical_cancer.csv')

FEATURE_NAMES = [
    'Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies',
    'Smokes', 'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives',
    'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)',
    'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
    'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
    'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
    'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',
    'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx'
]


class ModelManager:
    def __init__(self, dataset_path=DATASET_FILE, random_state=42):
        self.dataset_path = dataset_path
        self.random_state = random_state
        self.imputer = None
        self.scaler = None
        self.models = {}
        self.metrics = []
        self.feature_medians = {}
        self.image_models = {}
        self._is_initialized = False

    def initialize(self):
        """Carga datos, entrena el pipeline y calcula las métricas del benchmark."""
        if self._is_initialized:
            return

        df = pd.read_csv(self.dataset_path)
        df = df.replace('?', np.nan)
        
        # Eliminar variables de diagnóstico concomitante para evitar data leakage
        leakage_cols = ['Hinselmann', 'Schiller', 'Citology']
        df = df.drop(columns=leakage_cols, errors='ignore')

        target_col = 'Biopsy'
        X = df.drop(columns=[target_col]).astype(float)
        y = df[target_col].astype(int)

        # Partición estratificada (85% train, 15% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=self.random_state
        )

        # Imputación por mediana (ajustada en train)
        self.imputer = SimpleImputer(strategy='median')
        X_train_imp = self.imputer.fit_transform(X_train)
        X_test_imp = self.imputer.transform(X_test)

        # Guardar medianas para autocompletar valores faltantes en inferencia
        for col, med in zip(X.columns, self.imputer.statistics_):
            self.feature_medians[col] = float(med)

        # Escalado estándar (ajustado en train)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_imp)
        X_test_scaled = self.scaler.transform(X_test_imp)

        # Balanceo SMOTE sobre datos de entrenamiento
        smote = SMOTE(random_state=self.random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

        # Configuración de los 5 clasificadores
        self.models = {
            'Regresión Logística': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'SVM': SVC(
                probability=True, kernel='rbf', random_state=self.random_state
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            'XGBoost': XGBClassifier(
                eval_metric='logloss', random_state=self.random_state
            )
        }

        self.metrics = []
        for name, model in self.models.items():
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sensitivity = recall_score(y_test, y_pred, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = precision_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc_roc = roc_auc_score(y_test, y_proba)

            self.metrics.append({
                'model': name,
                'dataset': 'UCI',
                'accuracy': round(float(accuracy), 4),
                'accuracy_pct': f"{accuracy * 100:.2f}%",
                'sensitivity': round(float(sensitivity), 4),
                'sensitivity_pct': f"{sensitivity * 100:.2f}%",
                'specificity': round(float(specificity), 4),
                'specificity_pct': f"{specificity * 100:.2f}%",
                'precision': round(float(precision), 4),
                'precision_pct': f"{precision * 100:.2f}%",
                'f1': round(float(f1), 4),
                'f1_pct': f"{f1 * 100:.2f}%",
                'auc_roc': round(float(auc_roc), 4),
                'auc_roc_pct': f"{auc_roc * 100:.2f}%",
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            })

        # --- Agregar Métricas de los Modelos de Imagen (Fase 2) ---
        import json
        metrics_file = os.path.join(os.path.dirname(__file__), 'image_metrics.json')
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    image_metrics = json.load(f)
                for im in image_metrics:
                    def parse_metric(val):
                        if isinstance(val, str) and val.endswith('%'):
                            return float(val.strip('%')) / 100.0
                        return float(val) if val else 0.0
                        
                    acc = parse_metric(im.get('accuracy', 0))
                    sens = parse_metric(im.get('sensitivity', 0))
                    spec = parse_metric(im.get('specificity', 0))
                    prec = parse_metric(im.get('precision', 0))
                    f1_val = parse_metric(im.get('f1', 0))
                    auc_val = parse_metric(im.get('auc_roc', 0))
                    
                    self.metrics.append({
                        'model': im.get('model', 'MobileNet') + ' (CV)',
                        'dataset': im.get('dataset', 'Herlev'),
                        'accuracy': round(acc, 4), 'accuracy_pct': f"{acc * 100:.2f}%",
                        'sensitivity': round(sens, 4), 'sensitivity_pct': f"{sens * 100:.2f}%",
                        'specificity': round(spec, 4), 'specificity_pct': f"{spec * 100:.2f}%",
                        'precision': round(prec, 4), 'precision_pct': f"{prec * 100:.2f}%",
                        'f1': round(f1_val, 4), 'f1_pct': f"{f1_val * 100:.2f}%",
                        'auc_roc': round(auc_val, 4), 'auc_roc_pct': f"{auc_val * 100:.2f}%",
                        'tp': '-', 'fp': '-', 'tn': '-', 'fn': '-'
                    })
            except Exception as e:
                print(f"Error cargando métricas de imagen: {e}")
        else:
            # Placeholders if not trained yet
            datasets_cv = ['Herlev', 'SIPaKMeD', 'RIVA']
            models_cv = ['MobileNet', 'InceptionV3', 'ResNet50']
            for ds in datasets_cv:
                for m in models_cv:
                    self.metrics.append({
                        'model': m + ' (CV)',
                        'dataset': ds,
                        'accuracy': 0, 'accuracy_pct': "-",
                        'sensitivity': 0, 'sensitivity_pct': "-",
                        'specificity': 0, 'specificity_pct': "-",
                        'precision': 0, 'precision_pct': "-",
                        'f1': 0, 'f1_pct': "-",
                        'auc_roc': 0, 'auc_roc_pct': "-",
                        'tp': '-', 'fp': '-', 'tn': '-', 'fn': '-'
                    })

        self._is_initialized = True
        print("ModelManager inicializado correctamente.")

    def get_metrics(self):
        self.initialize()
        return self.metrics

    def predict(self, input_data: dict, model_name: str = 'Random Forest'):
        """
        Ejecuta la predicción para un conjunto de variables clínicas de una paciente.
        """
        self.initialize()

        if model_name not in self.models:
            model_name = 'Random Forest'

        # Construir vector de características en el orden estricto
        feature_vector = []
        for feature in FEATURE_NAMES:
            val = input_data.get(feature)
            if val is None or val == '' or (isinstance(val, str) and val.strip() == '?'):
                # Usar mediana de entrenamiento
                feature_vector.append(self.feature_medians[feature])
            else:
                try:
                    feature_vector.append(float(val))
                except (ValueError, TypeError):
                    feature_vector.append(self.feature_medians[feature])

        arr = np.array(feature_vector).reshape(1, -1)
        arr_scaled = self.scaler.transform(arr)

        # Predicción con el modelo seleccionado
        chosen_model = self.models[model_name]
        pred_class = int(chosen_model.predict(arr_scaled)[0])
        pred_proba = float(chosen_model.predict_proba(arr_scaled)[0][1])

        # Nivel de riesgo biomédico epidemiológico (Prevalencia basal en dataset = 6.4%)
        if pred_proba < 0.10:
            risk_tier = 'Bajo'
            risk_badge = 'safe'
            clinical_advice = 'Baja probabilidad (< 10%). Nivel acorde o inferior a la prevalencia poblacional basal (6.4%). Continuar cribado ginecológico regular.'
        elif pred_proba < 0.25:
            risk_tier = 'Moderado'
            risk_badge = 'warning'
            clinical_advice = 'Riesgo intermedio (10% - 25%). Riesgo 2 a 4 veces superior a la línea base poblacional. Se sugiere citología de control y prueba de ADN-VPH.'
        else:
            risk_tier = 'Alto'
            risk_badge = 'danger'
            clinical_advice = 'Alto riesgo (≥ 25%). Riesgo más de 4 veces superior al promedio poblacional. Indicación prioritaria de colposcopia y biopsia confirmatoria.'

        # Consenso de todos los 5 modelos para comparación inmediata
        consensus = {}
        for m_name, m_inst in self.models.items():
            p_cls = int(m_inst.predict(arr_scaled)[0])
            p_prob = float(m_inst.predict_proba(arr_scaled)[0][1])
            consensus[m_name] = {
                'prediction': p_cls,
                'probability': round(p_prob * 100, 1),
                'risk_tier': 'Alto' if p_prob >= 0.25 else ('Moderado' if p_prob >= 0.10 else 'Bajo')
            }

        return {
            'selected_model': model_name,
            'prediction': pred_class,
            'probability': round(pred_proba * 100, 1),
            'risk_tier': risk_tier,
            'risk_badge': risk_badge,
            'clinical_advice': clinical_advice,
            'consensus': consensus,
            'features_used': dict(zip(FEATURE_NAMES, feature_vector))
        }

    def get_presets(self):
        """Casos de prueba predefinidos para la interfaz clínica."""
        self.initialize()
        return {
            'low_risk': {
                'title': 'Paciente de Bajo Riesgo',
                'description': 'Mujer joven, hábitos saludables, sin historial de ITS ni diagnósticos previos.',
                'data': {
                    'Age': 22,
                    'Number of sexual partners': 1,
                    'First sexual intercourse': 18,
                    'Num of pregnancies': 0,
                    'Smokes': 0,
                    'Smokes (years)': 0,
                    'Smokes (packs/year)': 0,
                    'Hormonal Contraceptives': 1,
                    'Hormonal Contraceptives (years)': 1,
                    'IUD': 0,
                    'IUD (years)': 0,
                    'STDs': 0,
                    'STDs (number)': 0,
                    'Dx:Cancer': 0,
                    'Dx:CIN': 0,
                    'Dx:HPV': 0,
                    'Dx': 0
                }
            },
            'moderate_risk': {
                'title': 'Paciente de Riesgo Moderado',
                'description': 'Mujer adulta con tabaquismo prolongado y uso extendido de anticonceptivos hormonales.',
                'data': {
                    'Age': 38,
                    'Number of sexual partners': 4,
                    'First sexual intercourse': 16,
                    'Num of pregnancies': 3,
                    'Smokes': 1,
                    'Smokes (years)': 12,
                    'Smokes (packs/year)': 6,
                    'Hormonal Contraceptives': 1,
                    'Hormonal Contraceptives (years)': 7,
                    'IUD': 0,
                    'IUD (years)': 0,
                    'STDs': 0,
                    'STDs (number)': 0,
                    'Dx:Cancer': 0,
                    'Dx:CIN': 0,
                    'Dx:HPV': 0,
                    'Dx': 0
                }
            },
            'high_risk': {
                'title': 'Paciente de Alto Riesgo',
                'description': 'Mujer con historial de ITS (condilomatosis / VPH), tabaquismo y diagnóstico previo.',
                'data': {
                    'Age': 45,
                    'Number of sexual partners': 6,
                    'First sexual intercourse': 14,
                    'Num of pregnancies': 4,
                    'Smokes': 1,
                    'Smokes (years)': 20,
                    'Smokes (packs/year)': 15,
                    'Hormonal Contraceptives': 1,
                    'Hormonal Contraceptives (years)': 10,
                    'IUD': 1,
                    'IUD (years)': 5,
                    'STDs': 1,
                    'STDs (number)': 2,
                    'STDs:condylomatosis': 1,
                    'STDs:HPV': 1,
                    'Dx:Cancer': 0,
                    'Dx:CIN': 1,
                    'Dx:HPV': 1,
                    'Dx': 1
                }
            }
        }

    def get_available_image_models(self):
        """Retorna la lista de modelos de imagen disponibles y cargados."""
        models_dir = os.path.join(os.path.dirname(self.dataset_path), 'models')
        available = []
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('_herlev.keras'):
                    available.append(f.split('_')[0].capitalize())
        return available

    def predict_image(self, image_stream, model_name: str):
        """
        Realiza la inferencia para una imagen de citología usando un modelo pre-entrenado.
        Agrupa las clases en Normal (Bajo Riesgo) y Anormal (Alto Riesgo).
        """
        import tensorflow as tf
        
        # Validar y cargar modelo
        models_dir = os.path.join(os.path.dirname(self.dataset_path), 'models')
        model_path = os.path.join(models_dir, f"{model_name.lower()}_herlev.keras")
        
        if not os.path.exists(model_path):
            raise ValueError(f"El modelo {model_name} no se encuentra entrenado o disponible.")
            
        if model_name not in self.image_models:
            self.image_models[model_name] = tf.keras.models.load_model(model_path)
            
        img_model = self.image_models[model_name]
        
        # Preprocesar imagen
        img = Image.open(image_stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Inferir
        preds = img_model.predict(img_array)[0]
        predicted_class_idx = np.argmax(preds)
        confidence = float(preds[predicted_class_idx])
        
        # Mapeo de clases (Asumiendo orden alfabético de flow_from_directory)
        class_mapping = {
            0: ('carcinoma_in_situ', 'Anormal'),
            1: ('light_dysplastic', 'Anormal'),
            2: ('moderate_dysplastic', 'Anormal'),
            3: ('normal_columnar', 'Normal'),
            4: ('normal_intermediate', 'Normal'),
            5: ('normal_superficiel', 'Normal'),
            6: ('severe_dysplastic', 'Anormal')
        }
        
        class_name, group = class_mapping.get(predicted_class_idx, ('Desconocido', 'Desconocido'))
        
        # Consenso de todos los modelos disponibles (opcional)
        consensus = {}
        for avail_model in self.get_available_image_models():
            try:
                m_path = os.path.join(models_dir, f"{avail_model.lower()}_herlev.keras")
                if avail_model not in self.image_models:
                    self.image_models[avail_model] = tf.keras.models.load_model(m_path)
                m_preds = self.image_models[avail_model].predict(img_array)[0]
                m_idx = np.argmax(m_preds)
                m_prob = float(m_preds[m_idx])
                m_name, m_group = class_mapping.get(m_idx, ('Desconocido', 'Desconocido'))
                consensus[avail_model] = {
                    'prediction': m_group,
                    'specific_class': m_name,
                    'probability': round(m_prob * 100, 1),
                    'risk_tier': 'Alto' if m_group == 'Anormal' else 'Bajo'
                }
            except:
                pass
        
        if group == 'Anormal':
            risk_tier = 'Alto'
            risk_badge = 'danger'
            clinical_advice = f'Posible displasia o carcinoma ({class_name}). Se requiere evaluación colposcópica inmediata.'
        else:
            risk_tier = 'Bajo'
            risk_badge = 'safe'
            clinical_advice = f'Células de aspecto benigno ({class_name}). Continuar esquema de prevención regular.'

        return {
            'selected_model': model_name,
            'prediction': group,
            'specific_class': class_name,
            'probability': round(confidence * 100, 1),
            'risk_tier': risk_tier,
            'risk_badge': risk_badge,
            'clinical_advice': clinical_advice,
            'consensus': consensus
        }

# Instancia única reutilizable
manager = ModelManager()
