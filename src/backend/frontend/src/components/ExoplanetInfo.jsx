import React from 'react'
import { CheckCircle, AlertCircle, HelpCircle, Star, Orbit, Activity } from 'lucide-react'
import './ExoplanetInfo.css'

const ExoplanetInfo = ({ exoplanet }) => {
  const getStatusIcon = (status) => {
    switch (status) {
      case 'CONFIRMED':
        return <CheckCircle className="status-icon confirmed" size={20} />
      case 'FALSE_POSITIVE':
        return <AlertCircle className="status-icon false-positive" size={20} />
      default:
        return <HelpCircle className="status-icon candidate" size={20} />
    }
  }

  const getStatusLabel = (status) => {
    switch (status) {
      case 'CONFIRMED':
        return 'Confirmed'
      case 'FALSE_POSITIVE':
        return 'False Positive'
      default:
        return 'Candidate'
    }
  }

  return (
    <div className="exoplanet-info">
      {exoplanet.prediction && (
        <div className="info-section prediction-section">
          <h3 className="info-title">
            <Activity size={18} />
            AI Analysis
          </h3>
          <div className="prediction-result">
            <div className="prediction-status">
              {getStatusIcon(exoplanet.prediction)}
              <span className="prediction-label">{getStatusLabel(exoplanet.prediction)}</span>
            </div>
            <div className="confidence-display">
              <span className="confidence-text">Confidence</span>
              <span className="confidence-percentage">
                {(exoplanet.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="confidence-bar-full">
              <div
                className="confidence-bar-fill"
                style={{ width: `${exoplanet.confidence * 100}%` }}
              />
            </div>
          </div>

          {exoplanet.probabilities && (
            <div className="probabilities">
              <h4 className="probabilities-title">Probabilities</h4>
              {Object.entries(exoplanet.probabilities).map(([key, value]) => (
                <div key={key} className="probability-item">
                  <span className="probability-label">{getStatusLabel(key)}</span>
                  <span className="probability-value">{(value * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {exoplanet.stellar_params && (
        <div className="info-section">
          <h3 className="info-title">
            <Star size={18} />
            Stellar Parameters
          </h3>
          <div className="info-grid">
            {exoplanet.stellar_params.teff && (
              <div className="info-item">
                <span className="info-label">Temperature</span>
                <span className="info-value">{exoplanet.stellar_params.teff.toFixed(0)} K</span>
              </div>
            )}
            {exoplanet.stellar_params.radius && (
              <div className="info-item">
                <span className="info-label">Radius</span>
                <span className="info-value">{exoplanet.stellar_params.radius.toFixed(2)} R☉</span>
              </div>
            )}
            {exoplanet.stellar_params.mass && (
              <div className="info-item">
                <span className="info-label">Mass</span>
                <span className="info-value">{exoplanet.stellar_params.mass.toFixed(2)} M☉</span>
              </div>
            )}
            {exoplanet.stellar_params.logg && (
              <div className="info-item">
                <span className="info-label">log g</span>
                <span className="info-value">{exoplanet.stellar_params.logg.toFixed(2)}</span>
              </div>
            )}
            {exoplanet.stellar_params.feh !== undefined && (
              <div className="info-item">
                <span className="info-label">Metalicidade</span>
                <span className="info-value">{exoplanet.stellar_params.feh.toFixed(2)}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {exoplanet.transit_params && (
        <div className="info-section">
          <h3 className="info-title">
            <Orbit size={18} />
            Transit Parameters
          </h3>
          <div className="info-grid">
            {exoplanet.transit_params.period && (
              <div className="info-item">
                <span className="info-label">Orbital Period</span>
                <span className="info-value">{exoplanet.transit_params.period.toFixed(2)} days</span>
              </div>
            )}
            {exoplanet.transit_params.duration && (
              <div className="info-item">
                <span className="info-label">Duration</span>
                <span className="info-value">{exoplanet.transit_params.duration.toFixed(2)} hours</span>
              </div>
            )}
            {exoplanet.transit_params.depth && (
              <div className="info-item">
                <span className="info-label">Depth</span>
                <span className="info-value">{(exoplanet.transit_params.depth * 100).toFixed(4)}%</span>
              </div>
            )}
            {exoplanet.transit_params.epoch && (
              <div className="info-item">
                <span className="info-label">Epoch</span>
                <span className="info-value">{exoplanet.transit_params.epoch.toFixed(1)} BJD</span>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="info-section">
        <h3 className="info-title">General Information</h3>
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Mission</span>
            <span className="info-value">{exoplanet.mission}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Status</span>
            <span className="info-value">{getStatusLabel(exoplanet.status)}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ExoplanetInfo
