import React, { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react'
import './LightCurveChart.css'

const LightCurveChart = ({ data, targetName }) => {
  const [normalized, setNormalized] = useState(true)
  const [zoom, setZoom] = useState(1)

  if (!data || !data.flux || !data.time) {
    return <div className="chart-error">Light curve data not available</div>
  }

  // Prepare chart data
  const chartData = data.time.map((time, index) => ({
    time: time,
    flux: data.flux[index]
  }))

  const handleZoomIn = () => setZoom(prev => Math.min(prev * 1.5, 5))
  const handleZoomOut = () => setZoom(prev => Math.max(prev / 1.5, 1))
  const handleReset = () => {
    setZoom(1)
    setNormalized(true)
  }

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-label">Time: {payload[0].payload.time.toFixed(4)}</p>
          <p className="tooltip-value">Flux: {payload[0].value.toFixed(6)}</p>
        </div>
      )
    }
    return null
  }

  return (
    <div className="light-curve-chart">
      <div className="chart-controls">
        <div className="control-group">
          <label className="control-label">
            <input
              type="checkbox"
              checked={normalized}
              onChange={(e) => setNormalized(e.target.checked)}
              className="control-checkbox"
            />
            <span>Normalize Data</span>
          </label>
        </div>

        <div className="zoom-controls">
          <button
            className="zoom-button"
            onClick={handleZoomOut}
            disabled={zoom <= 1}
            aria-label="Zoom out"
            title="Zoom out"
          >
            <ZoomOut size={18} />
          </button>
          <span className="zoom-level">{(zoom * 100).toFixed(0)}%</span>
          <button
            className="zoom-button"
            onClick={handleZoomIn}
            disabled={zoom >= 5}
            aria-label="Zoom in"
            title="Zoom in"
          >
            <ZoomIn size={18} />
          </button>
          <button
            className="zoom-button"
            onClick={handleReset}
            aria-label="Reset view"
            title="Reset"
          >
            <RotateCcw size={18} />
          </button>
        </div>
      </div>

      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="time"
              stroke="#94a3b8"
              label={{ value: 'Time (days)', position: 'insideBottom', offset: -10, fill: '#94a3b8' }}
              tickFormatter={(value) => value.toFixed(2)}
            />
            <YAxis
              stroke="#94a3b8"
              label={{ value: 'Normalized Flux', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
              tickFormatter={(value) => value.toFixed(4)}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={1} stroke="#475569" strokeDasharray="3 3" />
            <Line
              type="monotone"
              dataKey="flux"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6, fill: '#3b82f6' }}
              isAnimationActive={true}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-description">
        <p>
          <strong>Interpretation:</strong> The light curve shows the variation of the star's brightness over time.
          Periodic dips in brightness indicate potential planetary transits.
        </p>
      </div>
    </div>
  )
}

export default LightCurveChart
