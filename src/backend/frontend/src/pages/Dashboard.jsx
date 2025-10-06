import React, { useState, useEffect } from 'react'
import { Search, Filter, Loader, AlertCircle } from 'lucide-react'
import SearchBar from '../components/SearchBar'
import FilterPanel from '../components/FilterPanel'
import ExoplanetCard from '../components/ExoplanetCard'
import StatsOverview from '../components/StatsOverview'
import { searchCatalog, getSystemMetrics } from '../services/api'
import './Dashboard.css'

const Dashboard = () => {
  const [exoplanets, setExoplanets] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [filters, setFilters] = useState({
    mission: '',
    method: 'Transit',
    status: '',
    starType: '',
    page: 1,
    pageSize: 25
  })
  const [stats, setStats] = useState(null)
  const [showFilters, setShowFilters] = useState(false)
  const [totalResults, setTotalResults] = useState(0)

  useEffect(() => {
    fetchMetrics()
    loadCatalogData()
  }, [])

  useEffect(() => {
    loadCatalogData()
  }, [searchQuery, filters.mission, filters.status, filters.page])

  const fetchMetrics = async () => {
    try {
      const metrics = await getSystemMetrics()
      setStats(metrics)
    } catch (err) {
      console.error('Error loading metrics:', err)
    }
  }

  const loadCatalogData = async () => {
    setLoading(true)
    setError(null)

    try {
      console.log('Loading catalog data...', { searchQuery, filters })
      const response = await searchCatalog(searchQuery, filters)
      console.log('Catalog response:', response)

      // Transform API data to match our component structure
      const transformedData = response.items.map(item => ({
        target_name: item.target_name,
        mission: item.mission,
        status: item.status || 'CANDIDATE',
        stellar_params: {
          teff: item.stellar_params?.teff,
          radius: item.stellar_params?.radius,
          mass: item.stellar_params?.mass
        },
        transit_params: {
          period: item.orbital_period,
          depth: item.transit_depth
        },
        ids: item.ids,
        prediction: item.status,
        confidence: item.confidence || 0.85
      }))

      setExoplanets(transformedData)
      setTotalResults(response.total)
    } catch (err) {
      console.error('Error loading catalog data:', err)
      setError('Loading NASA catalog data... (first load may take a while)')
      // Fallback to sample data on error
      setTimeout(() => {
        loadSampleData()
      }, 1000)
    } finally {
      setLoading(false)
    }
  }

  const loadSampleData = () => {
    const sampleData = [
      {
        target_name: 'Kepler-186f',
        mission: 'Kepler',
        status: 'CONFIRMED',
        stellar_params: { teff: 3788, radius: 0.47, mass: 0.48 },
        transit_params: { period: 129.9, depth: 0.0012 },
        prediction: 'CONFIRMED',
        confidence: 0.95
      },
      {
        target_name: 'Kepler-452b',
        mission: 'Kepler',
        status: 'CONFIRMED',
        stellar_params: { teff: 5757, radius: 1.11, mass: 1.04 },
        transit_params: { period: 384.8, depth: 0.0008 },
        prediction: 'CONFIRMED',
        confidence: 0.92
      },
      {
        target_name: 'TOI-700d',
        mission: 'TESS',
        status: 'CONFIRMED',
        stellar_params: { teff: 3480, radius: 0.42, mass: 0.41 },
        transit_params: { period: 37.4, depth: 0.0015 },
        prediction: 'CONFIRMED',
        confidence: 0.88
      }
    ]
    setExoplanets(sampleData)
    setTotalResults(sampleData.length)
  }

  const handleSearch = async (query) => {
    setSearchQuery(query)
    setFilters(prev => ({ ...prev, page: 1 }))
  }

  const handleFilterChange = (newFilters) => {
    setFilters(prev => ({ ...prev, ...newFilters, page: 1 }))
  }

  const handlePageChange = (newPage) => {
    setFilters(prev => ({ ...prev, page: newPage }))
  }

  return (
    <div className="dashboard page-enter">
      <div className="dashboard-header">
        <div>
          <h1 className="dashboard-title">Exoplanet Detector</h1>
          <p className="dashboard-subtitle">
            Explore and analyze exoplanet candidates using Machine Learning
          </p>
        </div>
      </div>

      {stats && <StatsOverview stats={stats} />}

      <div className="dashboard-controls">
        <SearchBar onSearch={handleSearch} />
        <button
          className="filter-toggle"
          onClick={() => setShowFilters(!showFilters)}
          aria-expanded={showFilters}
          aria-label="Toggle filters"
        >
          <Filter size={18} />
          Filters
        </button>
      </div>

      {showFilters && (
        <FilterPanel filters={filters} onChange={handleFilterChange} />
      )}

      {loading ? (
        <div className="loading-state" role="status" aria-live="polite">
          <Loader className="animate-spin" size={40} aria-hidden="true" />
          <p>Loading NASA catalog data...</p>
        </div>
      ) : error ? (
        <div className="error-state" role="alert">
          <AlertCircle size={40} aria-hidden="true" />
          <p>{error}</p>
          <button className="btn-primary" onClick={loadCatalogData}>
            Try Again
          </button>
        </div>
      ) : (
        <>
          <div className="results-header">
            <h2 className="results-count">
              {totalResults} {totalResults === 1 ? 'result' : 'results'} found
            </h2>
          </div>

          <div className="exoplanet-grid">
            {exoplanets.map((planet) => (
              <ExoplanetCard key={planet.target_name} planet={planet} />
            ))}
          </div>

          {exoplanets.length === 0 && (
            <div className="empty-state">
              <Search size={48} aria-hidden="true" />
              <p>No results found</p>
              <p className="empty-subtitle">Try adjusting your filters or search term</p>
            </div>
          )}

          {totalResults > filters.pageSize && (
            <div className="pagination">
              <button
                className="pagination-btn"
                onClick={() => handlePageChange(filters.page - 1)}
                disabled={filters.page === 1}
              >
                Previous
              </button>
              <span className="pagination-info">
                Page {filters.page} of {Math.ceil(totalResults / filters.pageSize)}
              </span>
              <button
                className="pagination-btn"
                onClick={() => handlePageChange(filters.page + 1)}
                disabled={filters.page >= Math.ceil(totalResults / filters.pageSize)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default Dashboard
