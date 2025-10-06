import React from 'react'
import { Activity, TrendingUp, CheckCircle, AlertTriangle } from 'lucide-react'
import './StatsOverview.css'

const StatsOverview = ({ stats }) => {
  const modelMetrics = stats?.model_metrics || {}
  const apiMetrics = stats?.api_metrics || {}

  const statCards = [
    {
      label: 'Model Accuracy',
      value: modelMetrics.accuracy ? `${(modelMetrics.accuracy * 100).toFixed(1)}%` : 'N/A',
      icon: Activity,
      color: 'primary'
    },
    {
      label: 'Total Predictions',
      value: modelMetrics.total_predictions || 0,
      icon: TrendingUp,
      color: 'success'
    },
    {
      label: 'Success Rate',
      value: apiMetrics.error_rate ? `${((1 - apiMetrics.error_rate) * 100).toFixed(1)}%` : '100%',
      icon: CheckCircle,
      color: 'success'
    },
    {
      label: 'Average Time',
      value: modelMetrics.avg_processing_time ? `${(modelMetrics.avg_processing_time * 1000).toFixed(0)}ms` : 'N/A',
      icon: AlertTriangle,
      color: 'warning'
    }
  ]

  return (
    <div className="stats-overview">
      {statCards.map((stat, index) => (
        <div key={index} className={`stat-card stat-${stat.color}`}>
          <div className="stat-icon-wrapper">
            <stat.icon className="stat-icon" size={24} aria-hidden="true" />
          </div>
          <div className="stat-content">
            <p className="stat-label">{stat.label}</p>
            <p className="stat-value">{stat.value}</p>
          </div>
        </div>
      ))}
    </div>
  )
}

export default StatsOverview
