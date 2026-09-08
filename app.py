"""
Servidor Web Flask para la Aplicación Interactiva de Detección de Cáncer Cervicouterino.
"""

import os
from flask import Flask, render_template, request, jsonify
from model_service import manager, FEATURE_NAMES

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cervical-cancer-ml-secret'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

# Pre-inicializar el modelo al arrancar
manager.initialize()


@app.route('/')
def index():
    """Página principal de la aplicación."""
    metrics = manager.get_metrics()
    presets = manager.get_presets()
    return render_template(
        'index.html',
        metrics=metrics,
        presets=presets,
        feature_names=FEATURE_NAMES
    )


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Retorna las métricas comparativas de los 5 modelos."""
    return jsonify({
        'status': 'success',
        'metrics': manager.get_metrics()
    })


@app.route('/api/presets', methods=['GET'])
def api_presets():
    """Retorna los perfiles clínicos de ejemplo."""
    return jsonify({
        'status': 'success',
        'presets': manager.get_presets()
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Ejecuta la predicción para los datos recibidos y el modelo seleccionado.
    """
    try:
        data = request.get_json() or {}
        model_name = data.get('model', 'Random Forest')
        features = data.get('features', {})
        
        result = manager.predict(features, model_name=model_name)
        return jsonify({
            'status': 'success',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/api/predict_image', methods=['POST'])
def api_predict_image():
    """
    Ejecuta la predicción para una imagen usando el modelo seleccionado.
    """
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No se proporcionó imagen'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No se seleccionó imagen'}), 400
        
    model_name = request.form.get('model', 'MobileNet')
    
    try:
        result = manager.predict_image(file.stream, model_name)
        return jsonify({
            'status': 'success',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Iniciando servidor en http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
