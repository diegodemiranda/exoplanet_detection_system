// Exoplanet Detection System JavaScript

// Application Data
const appData = {
    modelMetrics: {
        accuracy: 0.961,
        precision: { CONFIRMED: 0.945, CANDIDATE: 0.889, FALSE_POSITIVE: 0.978 },
        recall: { CONFIRMED: 0.923, CANDIDATE: 0.856, FALSE_POSITIVE: 0.985 },
        f1Score: { CONFIRMED: 0.934, CANDIDATE: 0.872, FALSE_POSITIVE: 0.981 }
    },
    statistics: {
        totalAnalyses: 15420,
        confirmedPlanets: 2834,
        candidates: 4521,
        falsePositives: 8065
    },
    exampleData: {
        confirmedPlanet: {
            name: "Kepler-452b",
            flux: [1.0002, 0.9998, 0.9995, 1.0001, 0.9997, 0.9985, 0.9990, 1.0003, 0.9999, 1.0001, 0.9996, 0.9994, 0.9989, 0.9987, 0.9983, 0.9986, 0.9991, 0.9995, 1.0000, 1.0002, 1.0001, 0.9999, 0.9997, 0.9994, 0.9988, 0.9985, 0.9983, 0.9987, 0.9992, 0.9998],
            mission: "Kepler",
            period: 384.8,
            depth: 152
        },
        falsePositive: {
            name: "Random-Star-001",
            flux: [1.0023, 1.0015, 0.9987, 1.0034, 0.9992, 1.0018, 0.9996, 1.0007, 1.0011, 0.9989, 1.0025, 0.9983, 1.0029, 0.9994, 1.0006, 1.0021, 0.9978, 1.0033, 0.9997, 1.0012, 0.9985, 1.0024, 1.0003, 0.9991, 1.0016, 0.9988, 1.0031, 0.9995, 1.0009, 1.0002],
            mission: "TESS"
        }
    }
};

// DOM Elements
const navButtons = document.querySelectorAll('.nav__button');
const sections = document.querySelectorAll('.section');
const individualForm = document.getElementById('individual-form');
const analysisResults = document.getElementById('analysis-results');
const loadExampleConfirmed = document.getElementById('load-example-confirmed');
const loadExampleFalse = document.getElementById('load-example-false');
const uploadArea = document.getElementById('upload-area');
const csvUpload = document.getElementById('csv-upload');
const batchResults = document.getElementById('batch-results');

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeCharts();
    initializeFormHandlers();
    initializeFileUpload();
    populateStatistics();
});

// Navigation
function initializeNavigation() {
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetSection = button.dataset.section;
            navButtons.forEach(btn => btn.classList.remove('nav__button--active'));
            button.classList.add('nav__button--active');
            sections.forEach(section => {
                section.classList.remove('section--active');
                if (section.id === targetSection) {
                    section.classList.add('section--active');
                }
            });
        });
    });
}

// Statistics Population
function populateStatistics() {
    const stats = appData.statistics;
    document.getElementById('total-analyses').textContent = stats.totalAnalyses.toLocaleString('pt-BR');
    document.getElementById('confirmed-planets').textContent = stats.confirmedPlanets.toLocaleString('pt-BR');
    document.getElementById('candidates').textContent = stats.candidates.toLocaleString('pt-BR');
    document.getElementById('false-positives').textContent = stats.falsePositives.toLocaleString('pt-BR');
}

// Charts Initialization
function initializeCharts() {
    createDistributionChart();
}

function createDistributionChart() {
    const ctx = document.getElementById('distributionChart').getContext('2d');
    const stats = appData.statistics;
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Planetas Confirmados', 'Candidatos', 'Falsos Positivos'],
            datasets: [{
                data: [stats.confirmedPlanets, stats.candidates, stats.falsePositives],
                backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C'],
                borderColor: ['#33808d', '#cc9a6b', '#904030'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#f5f5f5', padding: 20, font: { size: 14 } } },
                title: { display: true, text: 'Distribuição de Classificações', color: '#f5f5f5', font: { size: 18, weight: 'bold' } }
            }
        }
    });
}

// Form Handlers
function initializeFormHandlers() {
    loadExampleConfirmed.addEventListener('click', () => { loadExampleData('confirmed'); });
    loadExampleFalse.addEventListener('click', () => { loadExampleData('false'); });
    individualForm.addEventListener('submit', handleIndividualAnalysis);
}

function loadExampleData(type) {
    const example = type === 'confirmed' ? appData.exampleData.confirmedPlanet : appData.exampleData.falsePositive;
    document.getElementById('target-name').value = example.name;
    document.getElementById('mission').value = example.mission;
    document.getElementById('light-curve').value = example.flux.join(', ');
    if (example.period) document.getElementById('period').value = example.period;
    if (example.depth) document.getElementById('depth').value = example.depth;
}

function handleIndividualAnalysis(event) {
    event.preventDefault();
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    analyzeBtn.classList.add('btn--loading');
    btnLoader.classList.remove('hidden');
    const formData = {
        name: document.getElementById('target-name').value,
        mission: document.getElementById('mission').value,
        lightCurve: document.getElementById('light-curve').value,
        period: document.getElementById('period').value,
        depth: document.getElementById('depth').value
    };
    setTimeout(() => {
        const result = simulateAIAnalysis(formData);
        displayAnalysisResult(result);
        analyzeBtn.classList.remove('btn--loading');
        btnLoader.classList.add('hidden');
    }, 2000);
}

function simulateAIAnalysis(formData) {
    const fluxValues = formData.lightCurve.split(',').map(val => parseFloat(val.trim()));
    const fluxVariation = Math.max(...fluxValues) - Math.min(...fluxValues);
    const meanFlux = fluxValues.reduce((a, b) => a + b, 0) / fluxValues.length;
    let classification, confidence;
    let probabilities = { CONFIRMED: 0, CANDIDATE: 0, FALSE_POSITIVE: 0 };
    if (fluxVariation < 0.002 && Math.abs(meanFlux - 1.0) < 0.001) {
        if (formData.name.includes('Kepler') || formData.period) {
            classification = 'CONFIRMED';
            confidence = 0.89 + Math.random() * 0.1;
            probabilities.CONFIRMED = confidence;
            probabilities.CANDIDATE = (1 - confidence) * 0.6;
            probabilities.FALSE_POSITIVE = (1 - confidence) * 0.4;
        } else {
            classification = 'CANDIDATE';
            confidence = 0.75 + Math.random() * 0.15;
            probabilities.CANDIDATE = confidence;
            probabilities.CONFIRMED = (1 - confidence) * 0.3;
            probabilities.FALSE_POSITIVE = (1 - confidence) * 0.7;
        }
    } else {
        classification = 'FALSE_POSITIVE';
        confidence = 0.85 + Math.random() * 0.1;
        probabilities.FALSE_POSITIVE = confidence;
        probabilities.CANDIDATE = (1 - confidence) * 0.4;
        probabilities.CONFIRMED = (1 - confidence) * 0.6;
    }
    const total = Object.values(probabilities).reduce((a, b) => a + b, 0);
    Object.keys(probabilities).forEach(key => { probabilities[key] /= total; });
    return { classification, confidence, probabilities, lightCurve: fluxValues, targetName: formData.name };
}

function displayAnalysisResult(result) {
    const classificationLabels = { CONFIRMED: 'PLANETA CONFIRMADO', CANDIDATE: 'CANDIDATO A PLANETA', FALSE_POSITIVE: 'FALSO POSITIVO' };
    const classificationClass = `result-classification--${result.classification.toLowerCase().replace('_', '-')}`;
    const resultHTML = `
        <div class="result-content">
            <div class="result-header">
                <h3>Resultados da Análise: ${result.targetName}</h3>
                <div class="result-classification ${classificationClass}">${classificationLabels[result.classification]}</div>
                <div class="result-confidence">Confiança: ${(result.confidence * 100).toFixed(1)}%</div>
            </div>
            <div class="probabilities">
                <h4>Probabilidades Detalhadas:</h4>
                ${Object.entries(result.probabilities).map(([key, value]) => `
                    <div class="probability-item">
                        <span class="probability-label">${classificationLabels[key]}:</span>
                        <span class="probability-value">${(value * 100).toFixed(1)}%</span>
                    </div>
                `).join('')}
            </div>
            <div class="chart-container" style="position: relative; height: 300px;">
                <canvas id="lightCurveChart"></canvas>
            </div>
        </div>
    `;
    analysisResults.innerHTML = resultHTML;
    setTimeout(() => { createLightCurveChart(result.lightCurve); }, 100);
}

function createLightCurveChart(fluxData) {
    const ctx = document.getElementById('lightCurveChart').getContext('2d');
    const labels = fluxData.map((_, index) => `T${index + 1}`);
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Fluxo Normalizado',
                data: fluxData,
                borderColor: '#1FB8CD',
                backgroundColor: 'rgba(31, 184, 205, 0.1)',
                borderWidth: 2,
                pointRadius: 3,
                pointBackgroundColor: '#1FB8CD',
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#f5f5f5' } },
                title: { display: true, text: 'Curva de Luz', color: '#f5f5f5', font: { size: 16, weight: 'bold' } }
            },
            scales: {
                x: { title: { display: true, text: 'Tempo', color: '#f5f5f5' }, ticks: { color: '#a7a9a9' }, grid: { color: 'rgba(167, 169, 169, 0.2)' } },
                y: { title: { display: true, text: 'Fluxo Relativo', color: '#f5f5f5' }, ticks: { color: '#a7a9a9' }, grid: { color: 'rgba(167, 169, 169, 0.2)' } }
            }
        }
    });
}

// File Upload Functionality
function initializeFileUpload() {
    uploadArea.addEventListener('click', () => { csvUpload.click(); });
    uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('upload-area--dragover'); });
    uploadArea.addEventListener('dragleave', () => { uploadArea.classList.remove('upload-area--dragover'); });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('upload-area--dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) { handleFileUpload(files[0]); }
    });
    csvUpload.addEventListener('change', (e) => { if (e.target.files.length > 0) { handleFileUpload(e.target.files[0]); } });
}

function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) { alert('Por favor, selecione um arquivo CSV válido.'); return; }
    const reader = new FileReader();
    reader.onload = function(e) { const csvData = e.target.result; processBatchAnalysis(csvData); };
    reader.readAsText(file);
}

function processBatchAnalysis(csvData) {
    const lines = csvData.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const results = [];
    for (let i = 1; i < lines.length && i <= 10; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        const rowData = {};
        headers.forEach((header, index) => { rowData[header.toLowerCase()] = values[index]; });
        const analysisResult = simulateAIAnalysis({
            name: rowData.nome || rowData.name || `Target-${i}`,
            mission: rowData.missao || rowData.mission || 'Unknown',
            lightCurve: rowData.curva_luz || rowData.light_curve || generateRandomFlux(),
            period: rowData.periodo || rowData.period || '',
            depth: rowData.profundidade || rowData.depth || ''
        });
        results.push({
            name: rowData.nome || rowData.name || `Target-${i}`,
            mission: rowData.missao || rowData.mission || 'Unknown',
            classification: analysisResult.classification,
            confidence: (analysisResult.confidence * 100).toFixed(1)
        });
    }
    displayBatchResults(results);
}

function generateRandomFlux() {
    const flux = [];
    const baseFlux = 1.0;
    for (let i = 0; i < 30; i++) { flux.push((baseFlux + (Math.random() - 0.5) * 0.01).toFixed(6)); }
    return flux.join(', ');
}

function displayBatchResults(results) {
    const classificationLabels = { CONFIRMED: 'Confirmado', CANDIDATE: 'Candidato', FALSE_POSITIVE: 'Falso Positivo' };
    const tableHTML = `
        <h3>Resultados da Análise em Lote</h3>
        <p>Processados ${results.length} candidatos:</p>
        <table class="batch-table">
            <thead><tr><th>Nome do Alvo</th><th>Missão</th><th>Classificação</th><th>Confiança</th></tr></thead>
            <tbody>
                ${results.map(result => `
                    <tr>
                        <td>${result.name}</td>
                        <td>${result.mission}</td>
                        <td>
                            <span class="status status--${result.classification.toLowerCase().replace('_', '-') === 'confirmed' ? 'success' : result.classification.toLowerCase().replace('_', '-') === 'candidate' ? 'warning' : 'error'}">
                                ${classificationLabels[result.classification]}
                            </span>
                        </td>
                        <td>${result.confidence}%</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
        <div style="margin-top: 20px;">
            <button class="btn btn--primary" onclick="downloadResults()">📊 Baixar Resultados (CSV)</button>
        </div>
    `;
    batchResults.innerHTML = tableHTML;
    batchResults.classList.add('show');
    window.batchResultsData = results;
}

function downloadResults() {
    if (!window.batchResultsData) return;
    const csv = ['Nome,Missao,Classificacao,Confianca', ...window.batchResultsData.map(result => `${result.name},${result.mission},${result.classification},${result.confidence}%`)].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'resultados_exoplanetas.csv';
    a.click();
    window.URL.revokeObjectURL(url);
}

// Utility & monitoring
function formatNumber(num) { return num.toLocaleString('pt-BR'); }
window.addEventListener('error', function(e) { console.error('Application error:', e.error); });
window.addEventListener('load', function() { console.log('Exoplanet Detection System loaded successfully'); });

