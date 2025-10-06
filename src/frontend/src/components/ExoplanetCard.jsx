import React from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, CheckCircle, AlertCircle, HelpCircle } from 'lucide-react'
import './ExoplanetCard.css'

const ExoplanetCard = ({ planet }) => {
  const getStatusIcon = (status) => {
    switch (status) {
      case 'CONFIRMED':
        return <CheckCircle size={16} className="status-icon confirmed" />
      case 'FALSE_POSITIVE':
        return <AlertCircle size={16} className="status-icon false-positive" />
      default:
        return <HelpCircle size={16} className="status-icon candidate" />
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

  const getConfidenceClass = (confidence) => {
    if (confidence >= 0.9) return 'high'
    if (confidence >= 0.7) return 'medium'
    return 'low'
  }

  return (
    <article className="exoplanet-card">
      <div className="card-header">
        <h3 className="card-title">{planet.target_name}</h3>
        <div className="card-status">
          {getStatusIcon(planet.status)}
          <span className="status-label">{getStatusLabel(planet.status)}</span>
        </div>
      </div>

      <div className="card-content">
        <div className="card-meta">
          <span className="meta-label">Mission:</span>
          <span className="meta-value">{planet.mission}</span>
        </div>

        {planet.stellar_params && (
          <>
            <div className="card-meta">
              <span className="meta-label">Temperature:</span>
              <span className="meta-value">{planet.stellar_params.teff?.toFixed(0)} K</span>
            </div>
            <div className="card-meta">
              <span className="meta-label">Stellar Radius:</span>
              <span className="meta-value">{planet.stellar_params.radius?.toFixed(2)} R☉</span>
            </div>
          </>
        )}

        {planet.transit_params && (
          <>
            <div className="card-meta">
              <span className="meta-label">Orbital Period:</span>
              <span className="meta-value">{planet.transit_params.period?.toFixed(2)} days</span>
            </div>
            <div className="card-meta">
              <span className="meta-label">Depth:</span>
              <span className="meta-value">{(planet.transit_params.depth * 100)?.toFixed(3)}%</span>
            </div>
          </>
        )}

        {planet.confidence && (
          <div className="confidence-bar">
            <div className="confidence-label">
              <span>Confidence</span>
              <span className={`confidence-value ${getConfidenceClass(planet.confidence)}`}>
                {(planet.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="confidence-track">
              <div
                className={`confidence-fill ${getConfidenceClass(planet.confidence)}`}
                style={{ width: `${planet.confidence * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>

      <Link
        to={`/exoplanet/${encodeURIComponent(planet.target_name)}`}
        className="card-link"
        aria-label={`View details of ${planet.target_name}`}
      >
        View Details
        <ArrowRight size={16} aria-hidden="true" />
      </Link>
    </article>
  )
}

export default ExoplanetCard
