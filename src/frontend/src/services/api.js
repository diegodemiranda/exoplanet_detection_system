import axios from 'axios'

// Configure axios instance - remove /api prefix for production
const api = axios.create({
  baseURL: window.location.origin, // Use current origin instead of /api
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // Increased to 60 seconds for catalog loading
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.message || error.message || 'Unknown error'
    console.error('API Error:', message)
    return Promise.reject(error)
  }
)

// API Methods
export const searchCatalog = async (query = '', filters = {}) => {
  try {
    const params = {
      query: query,
      mission: filters.mission ? [filters.mission] : undefined,
      status: filters.status ? [filters.status] : undefined,
      page: filters.page || 1,
      page_size: filters.pageSize || 25
    }

    const response = await api.get('/catalog/search', { params })
    return response.data
  } catch (error) {
    console.error('Search catalog error:', error)
    throw error
  }
}

export const getLightCurve = async (mission, ids, targetName) => {
  try {
    const params = {
      mission,
      target_name: targetName,
      download: true,
      ...ids
    }
    const response = await api.get('/lightcurve', { params })
    return response.data
  } catch (error) {
    console.error('Light curve error:', error)
    throw error
  }
}

export const predictExoplanet = async (candidateData) => {
  try {
    const response = await api.post('/predict', candidateData)
    return response.data
  } catch (error) {
    console.error('Prediction error:', error)
    throw error
  }
}

export const batchPredict = async (candidates) => {
  try {
    const response = await api.post('/predict/batch', { candidates })
    return response.data
  } catch (error) {
    throw error
  }
}

export const getSystemMetrics = async () => {
  try {
    const response = await api.get('/metrics/json')
    return response.data
  } catch (error) {
    console.warn('Failed to load metrics:', error)
    return null
  }
}

export const getHealthStatus = async () => {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    throw error
  }
}

export default api
