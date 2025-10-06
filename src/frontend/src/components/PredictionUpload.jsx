import React, { useState } from 'react'
import { Upload, FileText, Loader, CheckCircle, XCircle, Info } from 'lucide-react'
import { predictExoplanet, batchPredict } from '../services/api'
import './PredictionUpload.css'

const PredictionUpload = () => {
  const [activeTab, setActiveTab] = useState('manual')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  // Manual input state
  const [manualData, setManualData] = useState({
    target_name: '',
    flux_data: '',
    mission: 'Kepler',
    teff: '',
    logg: '',
    feh: '',
    radius: '',
    mass: '',
    period: '',
    epoch: '',
    duration: '',
    depth: ''
  })

  // CSV upload state
  const [csvFile, setCsvFile] = useState(null)
  const [csvResults, setCsvResults] = useState(null)

  const handleManualInputChange = (field, value) => {
    setManualData(prev => ({ ...prev, [field]: value }))
    setError(null)
  }

  const parseFluxData = (fluxString) => {
    try {
      const values = fluxString.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
      if (values.length < 100) {
        throw new Error('Flux data must have at least 100 points')
      }
      return values
    } catch (err) {
      throw new Error('Invalid format. Use comma-separated values')
    }
  }

  const handleManualSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const fluxArray = parseFluxData(manualData.flux_data)

      const candidate = {
        target_name: manualData.target_name || 'Custom-Target',
        light_curve: {
          flux: fluxArray,
          mission: manualData.mission
        },
        stellar_params: manualData.teff ? {
          teff: parseFloat(manualData.teff),
          logg: manualData.logg ? parseFloat(manualData.logg) : undefined,
          feh: manualData.feh ? parseFloat(manualData.feh) : undefined,
          radius: manualData.radius ? parseFloat(manualData.radius) : undefined,
          mass: manualData.mass ? parseFloat(manualData.mass) : undefined
        } : undefined,
        transit_params: manualData.period ? {
          period: parseFloat(manualData.period),
          epoch: manualData.epoch ? parseFloat(manualData.epoch) : undefined,
          duration: manualData.duration ? parseFloat(manualData.duration) : undefined,
          depth: manualData.depth ? parseFloat(manualData.depth) : undefined
        } : undefined
      }

      const prediction = await predictExoplanet(candidate)
      setResult(prediction)
    } catch (err) {
      setError(err.response?.data?.message || err.message || 'Error processing prediction')
    } finally {
      setLoading(false)
    }
  }

  const handleCsvUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setCsvFile(file)
    setError(null)
    setCsvResults(null)

    if (!file.name.endsWith('.csv')) {
      setError('Please select a CSV file')
      return
    }

    setLoading(true)

    try {
      const text = await file.text()
      const lines = text.split('\n').filter(line => line.trim())

      if (lines.length < 2) {
        throw new Error('CSV file must have a header and at least one data row')
      }

      const header = lines[0].split(',').map(h => h.trim())
      const candidates = []

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim())

        if (values.length < 2) continue

        const targetName = values[0]
        const fluxData = values.slice(1).map(v => parseFloat(v)).filter(v => !isNaN(v))

        if (fluxData.length >= 100) {
          candidates.push({
            target_name: targetName || `Target-${i}`,
            light_curve: {
              flux: fluxData,
              mission: 'Kepler'
            }
          })
        }
      }

      if (candidates.length === 0) {
        throw new Error('No valid candidates found in the CSV')
      }

      const results = await batchPredict(candidates)
      setCsvResults(results)
    } catch (err) {
      setError(err.response?.data?.message || err.message || 'Error processing CSV file')
    } finally {
      setLoading(false)
    }
  }

  const getResultIcon = (prediction) => {
    if (prediction === 'CONFIRMED') return <CheckCircle className="result-icon confirmed" size={48} />
    if (prediction === 'FALSE_POSITIVE') return <XCircle className="result-icon false-positive" size={48} />
    return <Info className="result-icon candidate" size={48} />
  }

  const getResultLabel = (prediction) => {
    if (prediction === 'CONFIRMED') return 'Confirmed Exoplanet'
    if (prediction === 'FALSE_POSITIVE') return 'False Positive'
    return 'Exoplanet Candidate'
  }

  return (
    <div className="prediction-upload">
      <div className="upload-header">
        <h2 className="upload-title">Custom Data Analysis</h2>
        <p className="upload-subtitle">
          Upload your own data or enter manually for AI analysis
        </p>
      </div>

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'manual' ? 'active' : ''}`}
          onClick={() => setActiveTab('manual')}
        >
          <FileText size={18} />
          Manual Input
        </button>
        <button
          className={`tab ${activeTab === 'csv' ? 'active' : ''}`}
          onClick={() => setActiveTab('csv')}
        >
          <Upload size={18} />
          CSV Upload
        </button>
      </div>

      {activeTab === 'manual' ? (
        <div className="tab-content">
          <form onSubmit={handleManualSubmit} className="manual-form">
            <div className="form-section">
              <h3 className="section-title">Basic Information</h3>
              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="target_name">Target Name</label>
                  <input
                    id="target_name"
                    type="text"
                    placeholder="e.g., My-Exoplanet-1"
                    value={manualData.target_name}
                    onChange={(e) => handleManualInputChange('target_name', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="mission">Mission</label>
                  <select
                    id="mission"
                    value={manualData.mission}
                    onChange={(e) => handleManualInputChange('mission', e.target.value)}
                  >
                    <option value="Kepler">Kepler</option>
                    <option value="TESS">TESS</option>
                    <option value="K2">K2</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="form-section">
              <h3 className="section-title">Flux Data (Required)</h3>
              <div className="form-group full-width">
                <label htmlFor="flux_data">
                  Normalized Flux Data (min 100 points, comma-separated)
                </label>
                <textarea
                  id="flux_data"
                  placeholder="e.g., 1.0002, 0.9998, 0.9995, 1.0001, ..."
                  rows={6}
                  value={manualData.flux_data}
                  onChange={(e) => handleManualInputChange('flux_data', e.target.value)}
                  required
                />
                <small className="form-hint">
                  Paste your comma-separated flux values. Minimum 100 points.
                </small>
              </div>
            </div>

            <div className="form-section">
              <h3 className="section-title">Stellar Parameters (Optional)</h3>
              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="teff">Effective Temperature (K)</label>
                  <input
                    id="teff"
                    type="number"
                    step="0.01"
                    placeholder="e.g., 5777"
                    value={manualData.teff}
                    onChange={(e) => handleManualInputChange('teff', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="logg">log g (cm/s²)</label>
                  <input
                    id="logg"
                    type="number"
                    step="0.01"
                    placeholder="e.g., 4.44"
                    value={manualData.logg}
                    onChange={(e) => handleManualInputChange('logg', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="feh">Metallicity [Fe/H]</label>
                  <input
                    id="feh"
                    type="number"
                    step="0.01"
                    placeholder="e.g., 0.0"
                    value={manualData.feh}
                    onChange={(e) => handleManualInputChange('feh', e.target.value)}
                  />
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="radius">Stellar Radius (R☉)</label>
                  <input
                    id="radius"
                    type="number"
                    step="0.01"
                    placeholder="e.g., 1.0"
                    value={manualData.radius}
                    onChange={(e) => handleManualInputChange('radius', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="mass">Stellar Mass (M☉)</label>
                  <input
                    id="mass"
                    type="number"
                    step="0.01"
                    placeholder="e.g., 1.0"
                    value={manualData.mass}
                    onChange={(e) => handleManualInputChange('mass', e.target.value)}
                  />
                </div>
              </div>
            </div>

            <div className="form-section">
              <h3 className="section-title">Transit Parameters (Optional)</h3>
              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="period">Orbital Period (days)</label>
                  <input
                    id="period"
                    type="number"
                    step="0.01"
                    placeholder="e.g., 365.25"
                    value={manualData.period}
                    onChange={(e) => handleManualInputChange('period', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="epoch">Epoch (BJD)</label>
                  <input
                    id="epoch"
                    type="number"
                    step="0.01"
                    placeholder="e.g., 2455000"
                    value={manualData.epoch}
                    onChange={(e) => handleManualInputChange('epoch', e.target.value)}
                  />
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="duration">Duration (hours)</label>
                  <input
                    id="duration"
                    type="number"
                    step="0.01"
                    placeholder="e.g., 13.0"
                    value={manualData.duration}
                    onChange={(e) => handleManualInputChange('duration', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="depth">Depth (fraction)</label>
                  <input
                    id="depth"
                    type="number"
                    step="0.00001"
                    placeholder="e.g., 0.00084"
                    value={manualData.depth}
                    onChange={(e) => handleManualInputChange('depth', e.target.value)}
                  />
                </div>
              </div>
            </div>

            <button type="submit" className="submit-button" disabled={loading}>
              {loading ? (
                <>
                  <Loader className="animate-spin" size={18} />
                  Analyzing...
                </>
              ) : (
                'Analyze with AI'
              )}
            </button>
          </form>
        </div>
      ) : (
        <div className="tab-content">
          <div className="csv-upload-area">
            <div className="upload-instructions">
              <h3>CSV Format</h3>
              <p>The CSV file must contain:</p>
              <ul>
                <li>First column: <code>target_name</code> (target name)</li>
                <li>Following columns: flux values (minimum 100 points)</li>
                <li>Example: <code>Kepler-1,1.0002,0.9998,0.9995,...</code></li>
              </ul>
            </div>

            <div className="upload-box">
              <input
                type="file"
                id="csv-upload"
                accept=".csv"
                onChange={handleCsvUpload}
                className="file-input"
              />
              <label htmlFor="csv-upload" className="upload-label">
                <Upload size={48} />
                <span className="upload-text">
                  {csvFile ? csvFile.name : 'Click to select or drag a CSV file'}
                </span>
                <span className="upload-hint">CSV file with light curve data</span>
              </label>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="alert alert-error">
          <XCircle size={20} />
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className="prediction-result">
          <div className="result-header">
            {getResultIcon(result.prediction)}
            <h3 className="result-title">{getResultLabel(result.prediction)}</h3>
          </div>
          <div className="result-details">
            <div className="result-item">
              <span className="result-label">Alvo:</span>
              <span className="result-value">{result.target_name}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Confiança:</span>
              <span className="result-value confidence-value">
                {(result.confidence * 100).toFixed(2)}%
              </span>
            </div>
            {result.probabilities && (
              <div className="probabilities-grid">
                <h4>Probabilidades:</h4>
                {Object.entries(result.probabilities).map(([key, value]) => (
                  <div key={key} className="probability-item">
                    <span>{key}:</span>
                    <span>{(value * 100).toFixed(2)}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {csvResults && (
        <div className="csv-results">
          <h3 className="results-title">
            Resultados do Lote ({csvResults.results?.length || 0} predições)
          </h3>
          <div className="results-grid">
            {csvResults.results?.map((result, index) => (
              <div key={index} className="result-card">
                <div className="card-header">
                  {getResultIcon(result.prediction)}
                  <h4>{result.target_name}</h4>
                </div>
                <div className="card-body">
                  <div className="card-item">
                    <span>Predição:</span>
                    <span className={`prediction-badge ${result.prediction.toLowerCase()}`}>
                      {getResultLabel(result.prediction)}
                    </span>
                  </div>
                  <div className="card-item">
                    <span>Confiança:</span>
                    <span className="confidence-value">
                      {(result.confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
          {csvResults.batch_stats && (
            <div className="batch-stats">
              <h4>Estatísticas do Lote:</h4>
              <div className="stats-grid">
                <div className="stat-item">
                  <span>Total:</span>
                  <span>{csvResults.batch_stats.total_candidates}</span>
                </div>
                <div className="stat-item">
                  <span>Taxa de Sucesso:</span>
                  <span>{(csvResults.batch_stats.success_rate * 100).toFixed(1)}%</span>
                </div>
                <div className="stat-item">
                  <span>Tempo Médio:</span>
                  <span>{(csvResults.batch_stats.avg_processing_time * 1000).toFixed(0)}ms</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default PredictionUpload
