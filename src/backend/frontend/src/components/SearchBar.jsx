import React, { useState } from 'react'
import { Search } from 'lucide-react'
import './SearchBar.css'

const SearchBar = ({ onSearch }) => {
  const [query, setQuery] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    onSearch(query)
  }

  return (
    <form className="search-bar" onSubmit={handleSubmit} role="search">
      <label htmlFor="search-input" className="sr-only">
        Search exoplanets
      </label>
      <div className="search-input-wrapper">
        <Search className="search-icon" size={20} aria-hidden="true" />
        <input
          id="search-input"
          type="text"
          className="search-input"
          placeholder="Search by star or exoplanet name..."
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)
            onSearch(e.target.value)
          }}
          aria-label="Search field"
        />
      </div>
    </form>
  )
}

export default SearchBar
