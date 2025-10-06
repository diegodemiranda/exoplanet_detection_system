import React from 'react'
import './FilterPanel.css'

const FilterPanel = ({ filters, onChange }) => {
  const handleChange = (key, value) => {
    onChange({ ...filters, [key]: value })
  }

  return (
    <div className="filter-panel">
      <div className="filter-group">
        <label htmlFor="mission-filter" className="filter-label">
          Mission
        </label>
        <select
          id="mission-filter"
          className="filter-select"
          value={filters.mission}
          onChange={(e) => handleChange('mission', e.target.value)}
        >
          <option value="">All</option>
          <option value="Kepler">Kepler</option>
          <option value="K2">K2</option>
          <option value="TESS">TESS</option>
        </select>
      </div>

      <div className="filter-group">
        <label htmlFor="method-filter" className="filter-label">
          Detection Method
        </label>
        <select
          id="method-filter"
          className="filter-select"
          value={filters.method}
          onChange={(e) => handleChange('method', e.target.value)}
        >
          <option value="Transit">Transit</option>
          <option value="Radial Velocity">Radial Velocity</option>
          <option value="Direct Imaging">Direct Imaging</option>
          <option value="Microlensing">Microlensing</option>
        </select>
      </div>

      <div className="filter-group">
        <label htmlFor="status-filter" className="filter-label">
          Status
        </label>
        <select
          id="status-filter"
          className="filter-select"
          value={filters.status}
          onChange={(e) => handleChange('status', e.target.value)}
        >
          <option value="">All</option>
          <option value="CONFIRMED">Confirmed</option>
          <option value="CANDIDATE">Candidate</option>
          <option value="FALSE_POSITIVE">False Positive</option>
        </select>
      </div>

      <div className="filter-group">
        <label htmlFor="star-type-filter" className="filter-label">
          Star Type
        </label>
        <select
          id="star-type-filter"
          className="filter-select"
          value={filters.starType}
          onChange={(e) => handleChange('starType', e.target.value)}
        >
          <option value="">All</option>
          <option value="M">Type M (Red Dwarf)</option>
          <option value="K">Type K</option>
          <option value="G">Type G (Solar)</option>
          <option value="F">Type F</option>
          <option value="A">Type A</option>
        </select>
      </div>
    </div>
  )
}

export default FilterPanel
