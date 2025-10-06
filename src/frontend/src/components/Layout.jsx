import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Rocket, Home, Upload, Info } from 'lucide-react'
import './Layout.css'

const Layout = ({ children }) => {
  const location = useLocation()

  const isActive = (path) => location.pathname === path

  return (
    <div className="layout">
      <header className="header" role="banner">
        <div className="container">
          <div className="header-content">
            <Link to="/" className="logo" aria-label="Home page">
              <Rocket className="logo-icon" aria-hidden="true" />
              <span className="logo-text">Exoplanet Detector</span>
            </Link>

            <nav className="nav" role="navigation" aria-label="Main navigation">
              <Link
                to="/"
                className={`nav-link ${isActive('/') ? 'active' : ''}`}
                aria-current={isActive('/') ? 'page' : undefined}
              >
                <Home size={18} aria-hidden="true" />
                <span>Dashboard</span>
              </Link>
              <Link
                to="/analyze"
                className={`nav-link ${isActive('/analyze') ? 'active' : ''}`}
                aria-current={isActive('/analyze') ? 'page' : undefined}
              >
                <Upload size={18} aria-hidden="true" />
                <span>Analyze</span>
              </Link>
              <Link
                to="/about"
                className={`nav-link ${isActive('/about') ? 'active' : ''}`}
                aria-current={isActive('/about') ? 'page' : undefined}
              >
                <Info size={18} aria-hidden="true" />
                <span>About</span>
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <main className="main" role="main">
        <div className="container">
          {children}
        </div>
      </main>

      <footer className="footer" role="contentinfo">
        <div className="container">
          <p className="footer-text">
            &copy; {new Date().getFullYear()} NASA Space Apps Challenge - Exoplanet Detector
          </p>
          <p className="footer-text footer-credits">
            Built with React and Machine Learning
          </p>
        </div>
      </footer>
    </div>
  )
}

export default Layout
