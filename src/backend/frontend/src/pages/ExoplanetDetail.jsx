import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Loader, AlertCircle } from 'lucide-react'
import LightCurveChart from '../components/LightCurveChart'
import ExoplanetInfo from '../components/ExoplanetInfo'
import { predictExoplanet, getLightCurve } from '../services/api'
import './ExoplanetDetail.css'

const ExoplanetDetail = () => {
  const { targetName } = useParams()
  const navigate = useNavigate()
  const [exoplanet, setExoplanet] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [predicting, setPredicting] = useState(false)

  useEffect(() => {
    loadExoplanetData()
  }, [targetName])

  const loadExoplanetData = async () => {
    setLoading(true)
    setError(null)

    try {
      // For now, use sample data since we need to integrate with the catalog API
      // In a real implementation, you would fetch from /catalog/search with the specific target_name
      const sampleData = {
        target_name: decodeURIComponent(targetName),
        mission: 'Kepler',
        status: 'CONFIRMED',
        stellar_params: {
          teff: 5777,
          logg: 4.44,
          feh: 0.0,
          radius: 1.0,
          mass: 1.0
        },
        transit_params: {
          period: 365.25,
          epoch: 2455000.0,
          duration: 13.0,
          depth: 0.00084
        },
        light_curve: {
          time: Array.from({ length: 100 }, (_, i) => i * 0.01),
          flux: Array.from({ length: 100 }, (_, i) => {
            const transit = Math.abs(i - 50) < 5 ? -0.002 : 0
            return 1 + transit + (Math.random() - 0.5) * 0.0005
          }),
          mission: 'Kepler'
        },
        prediction: 'CONFIRMED',
        confidence: 0.95
      }

      setExoplanet(sampleData)
    } catch (err) {
      setError('Failed to load exoplanet data')
      console.error('Error loading exoplanet:', err)
    } finally {
      setLoading(false)
    }
  }

  const handlePredict = async () => {
    if (!exoplanet) return

    setPredicting(true)
    try {
      const result = await predictExoplanet({
        target_name: exoplanet.target_name,
        light_curve: exoplanet.light_curve,
        stellar_params: exoplanet.stellar_params,
        transit_params: exoplanet.transit_params
      })

      setExoplanet({
        ...exoplanet,
        prediction: result.prediction,
        confidence: result.confidence,
        probabilities: result.probabilities
      })
    } catch (err) {
      console.error('Prediction error:', err)
      alert('Prediction failed. Please try again.')
    } finally {
      setPredicting(false)
    }
  }

  if (loading) {
    return (
      <div className="detail-loading" role="status" aria-live="polite">
        <Loader className="animate-spin" size={40} aria-hidden="true" />
        <p>Loading exoplanet data...</p>
      </div>
    )
  }

  if (error || !exoplanet) {
    return (
      <div className="detail-error" role="alert">
        <AlertCircle size={40} aria-hidden="true" />
        <p>{error || 'Exoplanet not found'}</p>
        <button className="btn-primary" onClick={() => navigate('/')}>
          Back to Dashboard
        </button>
      </div>
    )
  }

  return (
    <div className="exoplanet-detail page-enter">
      <button
        className="back-button"
        onClick={() => navigate('/')}
        aria-label="Back to dashboard"
      >
        <ArrowLeft size={20} />
        Back
      </button>

      <div className="detail-header">
        <h1 className="detail-title">{exoplanet.target_name}</h1>
        <button
          className="btn-predict"
          onClick={handlePredict}
          disabled={predicting}
        >
          {predicting ? (
            <>
              <Loader className="animate-spin" size={16} />
              Analyzing...
            </>
          ) : (
            'Analyze with AI'
          )}
        </button>
      </div>

      <div className="detail-grid">
        <div className="detail-main">
          <section className="detail-section" aria-labelledby="light-curve-heading">
            <h2 id="light-curve-heading" className="section-title">
              Light Curve
            </h2>
            <LightCurveChart
              data={exoplanet.light_curve}
              targetName={exoplanet.target_name}
            />
          </section>
        </div>

        <div className="detail-sidebar">
          <ExoplanetInfo exoplanet={exoplanet} />
        </div>
      </div>
    </div>
  )
}

export default ExoplanetDetail
