/**
 * OncoPredict AI - Controlador de Interfaz y Cliente API
 */

let selectedModel = 'Random Forest';
let presetsCache = null;

document.addEventListener('DOMContentLoaded', async () => {
    // Cargar presets desde API
    try {
        const res = await fetch('/api/presets');
        const data = await res.json();
        if (data.status === 'success') {
            presetsCache = data.presets;
        }
    } catch (err) {
        console.warn('No se pudieron precargar los presets:', err);
    }

    // Inicializar inputs de tabaquismo
    toggleSmokingInputs();
});

/**
 * Cambia la pestaña activa (Inferencia vs Benchmark)
 */
function switchTab(tab) {
    const tabs = ['predict', 'benchmark', 'image'];
    tabs.forEach(t => {
        const btn = document.getElementById(`tab-btn-${t}`);
        const view = document.getElementById(`view-${t}`);
        if (btn && view) {
            if (t === tab) {
                btn.classList.add('active');
                btn.setAttribute('aria-selected', 'true');
                view.classList.add('active');
                view.style.display = 'block';
            } else {
                btn.classList.remove('active');
                btn.setAttribute('aria-selected', 'false');
                view.classList.remove('active');
                view.style.display = 'none';
            }
        }
    });
}

/**
 * Selecciona el modelo de Machine Learning activo
 */
function selectModel(modelName) {
    selectedModel = modelName;

    // Actualizar clases en las tarjetas
    const cards = document.querySelectorAll('.model-card');
    cards.forEach(c => {
        if (c.getAttribute('data-model') === modelName) {
            c.classList.add('active');
        } else {
            c.classList.remove('active');
        }
    });

    // Actualizar indicador en el panel de resultados
    const tag = document.getElementById('display-selected-model');
    if (tag) tag.textContent = modelName;

    // Si ya hay resultados visibles, re-ejecutar con el nuevo modelo para feedback instantáneo
    const content = document.getElementById('results-content');
    if (content && !content.classList.contains('hidden')) {
        runPrediction();
    }
}

/**
 * Habilita o deshabilita campos de tabaquismo según si la paciente fuma
 */
function toggleSmokingInputs() {
    const smokesSelect = document.getElementById('input-Smokes');
    const isSmoker = smokesSelect.value === '1';
    const yearsInput = document.getElementById('input-SmokesYears');
    const packsInput = document.getElementById('input-SmokesPacks');

    if (yearsInput && packsInput) {
        yearsInput.disabled = !isSmoker;
        packsInput.disabled = !isSmoker;
        if (!isSmoker) {
            yearsInput.value = '0';
            packsInput.value = '0';
        }
    }
}

/**
 * Carga un caso predefinido en el formulario
 */
function loadPreset(presetKey) {
    if (!presetsCache || !presetsCache[presetKey]) {
        console.warn('Preset no disponible en caché');
        return;
    }

    const preset = presetsCache[presetKey];
    const data = preset.data;

    // Llenar campos
    for (const [key, val] of Object.entries(data)) {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) {
            input.value = val;
        }
    }

    toggleSmokingInputs();

    // Ejecutar predicción automáticamente
    runPrediction();
}

/**
 * Limpia el formulario y regresa al estado inicial
 */
function resetForm() {
    const form = document.getElementById('patient-form');
    if (form) form.reset();
    toggleSmokingInputs();

    // Ocultar resultados
    document.getElementById('results-placeholder').classList.remove('hidden');
    document.getElementById('results-content').classList.add('hidden');
    
    // Ocultar estado en línea
    const inlineStatus = document.getElementById('inline-status-msg');
    if (inlineStatus) inlineStatus.classList.add('hidden');
}

/**
 * Desplaza suavemente la vista hacia el panel de resultados
 */
function scrollToResults() {
    const resultsCard = document.getElementById('results-card');
    if (resultsCard) {
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        resultsCard.classList.add('highlight-pulse');
        setTimeout(() => resultsCard.classList.remove('highlight-pulse'), 1500);
    }
}

/**
 * Envía los datos del formulario a la API y renderiza el dictamen clínico
 */
async function runPrediction() {
    const btnSubmit = document.getElementById('btn-submit-prediction');
    const btnText = document.getElementById('btn-submit-text');
    const originalText = btnText ? btnText.textContent : 'Ejecutar Análisis de Riesgo';

    btnSubmit.disabled = true;
    if (btnText) btnText.textContent = 'Procesando Inferencia...';

    // Recolectar datos del formulario de manera exhaustiva (incluso campos deshabilitados)
    const form = document.getElementById('patient-form');
    const inputs = form.querySelectorAll('input, select');
    const features = {};

    inputs.forEach(input => {
        if (input.name) {
            if (input.value !== '' && input.value !== null && input.value !== undefined) {
                const num = parseFloat(input.value);
                features[input.name] = isNaN(num) ? 0 : num;
            } else {
                features[input.name] = 0;
            }
        }
    });

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: selectedModel,
                features: features
            })
        });

        const data = await response.json();
        if (data.status === 'success') {
            renderResults(data.result);
            if (btnText) {
                btnText.textContent = '✓ ¡Análisis Calculado!';
                setTimeout(() => {
                    btnText.textContent = originalText;
                }, 2000);
            }
        } else {
            alert('Error en la predicción: ' + (data.message || 'Desconocido'));
            if (btnText) btnText.textContent = originalText;
        }
    } catch (err) {
        console.error('Error de red al ejecutar inferencia:', err);
        alert('Error al comunicar con el servidor de análisis.');
        if (btnText) btnText.textContent = originalText;
    } finally {
        btnSubmit.disabled = false;
    }
}

/**
 * Renderiza el dictamen, gauge animado y consenso de modelos
 */
function renderResults(result) {
    const placeholder = document.getElementById('results-placeholder');
    const content = document.getElementById('results-content');

    placeholder.classList.add('hidden');
    content.classList.remove('hidden');

    // Actualizar textos de veredicto
    const prob = result.probability; // Ej: 75.4
    document.getElementById('display-probability').textContent = `${prob}%`;
    document.getElementById('display-selected-model').textContent = result.selected_model;

    const verdictBox = document.getElementById('risk-verdict-box');
    const riskBadge = document.getElementById('display-risk-badge');
    const predTitle = document.getElementById('display-prediction-title');
    const clinicalAdvice = document.getElementById('display-clinical-advice');

    verdictBox.className = `risk-verdict-box ${result.risk_badge}`;
    riskBadge.textContent = `RIESGO ${result.risk_tier.toUpperCase()}`;
    predTitle.textContent = result.prediction === 1 ? 'Biopsia Positiva Sugerida' : 'Biopsia Negativa Probable';
    clinicalAdvice.textContent = result.clinical_advice;

    // Animar Gauge Circular (Perímetro r=50 es 2*pi*50 ≈ 314)
    const gaugeBar = document.getElementById('gauge-bar');
    const circumference = 314;
    const offset = circumference - (circumference * (prob / 100));
    gaugeBar.style.strokeDashoffset = offset;

    // Color del arco según nivel de riesgo
    let gaugeColor = '#10b981'; // safe
    if (result.risk_badge === 'warning') gaugeColor = '#f59e0b';
    if (result.risk_badge === 'danger') gaugeColor = '#f43f5e';
    gaugeBar.style.stroke = gaugeColor;

    // Renderizar consenso de todos los 5 modelos
    const consensusGrid = document.getElementById('consensus-grid');
    consensusGrid.innerHTML = '';

    if (result.consensus) {
        for (const [mName, mData] of Object.entries(result.consensus)) {
            const isSelected = mName === result.selected_model;
            const isHigh = mData.risk_tier === 'Alto';
            const riskClass = isHigh ? 'danger' : 'safe';

            const row = document.createElement('div');
            row.className = `consensus-row ${isSelected ? 'selected' : ''}`;
            row.innerHTML = `
                <div class="consensus-name">${mName} ${isSelected ? '<small>(activo)</small>' : ''}</div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: var(--text-muted); font-size: 0.75rem;">${mData.probability}%</span>
                    <span class="consensus-risk ${riskClass}">
                        ${isHigh ? 'Alto Riesgo' : 'Bajo Riesgo'}
                    </span>
                </div>
            `;
            consensusGrid.appendChild(row);
        }
    }

    // Actualizar barra de estado en línea debajo del botón
    const inlineBox = document.getElementById('inline-status-msg');
    const inlineBadge = document.getElementById('inline-status-desc');
    const inlineProb = document.getElementById('inline-status-prob');
    const inlineTitle = document.getElementById('inline-status-title');
    const inlineIcon = document.getElementById('inline-status-icon');

    if (inlineBox && inlineBadge && inlineProb) {
        inlineBox.className = `inline-status ${result.risk_badge}`;
        inlineBadge.textContent = `Riesgo ${result.risk_tier}`;
        inlineProb.textContent = `(${result.probability}%)`;
        inlineTitle.textContent = `Resultado con ${result.selected_model}:`;
        inlineIcon.textContent = result.risk_tier === 'Alto' ? '⚡' : (result.risk_tier === 'Moderado' ? '⚠' : '✓');
        inlineBox.classList.remove('hidden');
    }

    // Resaltar visualmente la tarjeta de resultados en el lateral
    const resultsCard = document.getElementById('results-card');
    if (resultsCard) {
        resultsCard.classList.add('highlight-pulse');
        setTimeout(() => resultsCard.classList.remove('highlight-pulse'), 1500);

        // Si la pantalla es menor a 1100px, hacer scroll automático
        if (window.innerWidth < 1100) {
            resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
}

// ----------------------------------------------------------------------
// Lógica para Análisis de Imágenes (Citología)
// ----------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    const imageInput = document.getElementById('image-input');
    if (imageInput) {
        imageInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('image-preview');
                    if (preview) {
                        preview.src = e.target.result;
                        const container = document.getElementById('image-preview-container');
                        if (container) container.classList.remove('hidden');
                        const label = document.querySelector('.upload-label');
                        if (label) label.style.display = 'none';
                    }
                }
                reader.readAsDataURL(file);
            }
        });
    }
});

function clearImageSelection() {
    const imageInput = document.getElementById('image-input');
    if (imageInput) imageInput.value = '';
    
    const container = document.getElementById('image-preview-container');
    if (container) container.classList.add('hidden');
    
    const preview = document.getElementById('image-preview');
    if (preview) preview.src = '';
    
    const label = document.querySelector('.upload-label');
    if (label) label.style.display = 'flex';
    
    const resPlaceholder = document.getElementById('image-results-placeholder');
    if (resPlaceholder) resPlaceholder.classList.remove('hidden');
    
    const resContent = document.getElementById('image-results-content');
    if (resContent) resContent.classList.add('hidden');
}

async function runImagePrediction() {
    const fileInput = document.getElementById('image-input');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        alert("Por favor selecciona una imagen primero.");
        return;
    }
    
    const file = fileInput.files[0];
    const model = document.getElementById('image-model-selector').value;
    
    const btnSubmit = document.getElementById('btn-submit-image');
    const btnText = document.getElementById('btn-submit-image-text');
    const originalText = btnText ? btnText.textContent : 'Analizar Imagen';
    
    if (btnSubmit) btnSubmit.disabled = true;
    if (btnText) btnText.textContent = 'Analizando...';
    
    const formData = new FormData();
    formData.append('image', file);
    formData.append('model', model);
    
    try {
        const response = await fetch('/api/predict_image', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            renderImageResults(data.result);
            if (btnText) {
                btnText.textContent = '✓ ¡Análisis Completado!';
                setTimeout(() => {
                    btnText.textContent = originalText;
                }, 2000);
            }
        } else {
            alert('Error en la predicción: ' + (data.message || 'Desconocido'));
            if (btnText) btnText.textContent = originalText;
        }
    } catch (err) {
        console.error('Error de red al ejecutar inferencia de imagen:', err);
        alert('Error al comunicar con el servidor.');
        if (btnText) btnText.textContent = originalText;
    } finally {
        if (btnSubmit) btnSubmit.disabled = false;
    }
}

function renderImageResults(result) {
    const resPlaceholder = document.getElementById('image-results-placeholder');
    if (resPlaceholder) resPlaceholder.classList.add('hidden');
    
    const resContent = document.getElementById('image-results-content');
    if (resContent) resContent.classList.remove('hidden');
    
    const probElem = document.getElementById('display-img-probability');
    if (probElem) probElem.textContent = `${result.probability}%`;
    
    const modelElem = document.getElementById('display-img-model');
    if (modelElem) modelElem.textContent = result.selected_model;
    
    const verdictBox = document.getElementById('img-risk-verdict-box');
    const riskBadge = document.getElementById('display-img-risk-badge');
    const predTitle = document.getElementById('display-img-prediction-title');
    const clinicalAdvice = document.getElementById('display-img-clinical-advice');
    
    if (verdictBox) verdictBox.className = `risk-verdict-box ${result.risk_badge}`;
    if (riskBadge) riskBadge.textContent = `RIESGO ${result.risk_tier.toUpperCase()}`;
    if (predTitle) predTitle.textContent = result.prediction;
    if (clinicalAdvice) clinicalAdvice.textContent = result.clinical_advice;
    
    const gaugeBar = document.getElementById('img-gauge-bar');
    if (gaugeBar) {
        const circumference = 314;
        const offset = circumference - (circumference * (result.probability / 100));
        gaugeBar.style.strokeDashoffset = offset;
        
        let gaugeColor = '#10b981'; // safe
        if (result.risk_badge === 'warning') gaugeColor = '#f59e0b';
        if (result.risk_badge === 'danger') gaugeColor = '#f43f5e';
        gaugeBar.style.stroke = gaugeColor;
    }
}
