import React from 'react'
import { Rocket, Brain, Zap, Shield, Github, Mail } from 'lucide-react'
import './About.css'

const About = () => {
  const features = [
    {
      icon: Brain,
      title: 'Machine Learning Avançado',
      description: 'Utiliza redes neurais profundas para detectar padrões de trânsito planetário com alta precisão.'
    },
    {
      icon: Zap,
      title: 'Performance Otimizada',
      description: 'Processamento rápido com cache inteligente e predições em tempo real.'
    },
    {
      icon: Shield,
      title: 'Dados Validados',
      description: 'Integração com dados oficiais das missões Kepler, K2 e TESS da NASA.'
    },
    {
      icon: Rocket,
      title: 'Visualização Interativa',
      description: 'Gráficos interativos de curvas de luz com zoom, pan e análise detalhada.'
    }
  ]

  const techStack = [
    { category: 'Frontend', items: ['React 18', 'Recharts', 'Vite', 'CSS3'] },
    { category: 'Backend', items: ['Python', 'FastAPI', 'TensorFlow/Keras', 'NumPy'] },
    { category: 'Machine Learning', items: ['CNN', 'LSTM', 'Attention Mechanism', 'Robust Scaling'] },
    { category: 'Data Sources', items: ['NASA Exoplanet Archive', 'MAST', 'Kepler/TESS Light Curves'] }
  ]

  return (
    <div className="about-page page-enter">
      <div className="about-hero">
        <h1 className="about-title">Sobre o Projeto</h1>
        <p className="about-subtitle">
          Uma plataforma moderna de detecção de exoplanetas usando Machine Learning e dados da NASA
        </p>
      </div>

      <section className="about-section" aria-labelledby="mission-heading">
        <h2 id="mission-heading" className="section-heading">Nossa Missão</h2>
        <p className="section-text">
          Este projeto foi desenvolvido para o <strong>NASA Space Apps Challenge</strong> com o objetivo
          de democratizar o acesso à análise de dados astronômicos. Utilizamos técnicas avançadas de
          aprendizado de máquina para identificar candidatos a exoplanetas a partir de curvas de luz
          das missões Kepler, K2 e TESS.
        </p>
        <p className="section-text">
          Nossa plataforma permite que estudantes, astrônomos amadores e entusiastas da ciência
          explorem e analisem dados de trânsito planetário de forma intuitiva e acessível.
        </p>
      </section>

      <section className="about-section" aria-labelledby="features-heading">
        <h2 id="features-heading" className="section-heading">Recursos Principais</h2>
        <div className="features-grid">
          {features.map((feature, index) => (
            <div key={index} className="feature-card">
              <div className="feature-icon">
                <feature.icon size={32} aria-hidden="true" />
              </div>
              <h3 className="feature-title">{feature.title}</h3>
              <p className="feature-description">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="about-section" aria-labelledby="tech-heading">
        <h2 id="tech-heading" className="section-heading">Tecnologias Utilizadas</h2>
        <div className="tech-grid">
          {techStack.map((tech, index) => (
            <div key={index} className="tech-card">
              <h3 className="tech-category">{tech.category}</h3>
              <ul className="tech-list">
                {tech.items.map((item, i) => (
                  <li key={i} className="tech-item">{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <section className="about-section" aria-labelledby="how-it-works-heading">
        <h2 id="how-it-works-heading" className="section-heading">Como Funciona</h2>
        <div className="workflow">
          <div className="workflow-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h3 className="step-title">Coleta de Dados</h3>
              <p className="step-description">
                Dados de curvas de luz são obtidos do NASA Exoplanet Archive e MAST.
              </p>
            </div>
          </div>
          <div className="workflow-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h3 className="step-title">Pré-processamento</h3>
              <p className="step-description">
                Normalização robusta, remoção de outliers e extração de características estatísticas.
              </p>
            </div>
          </div>
          <div className="workflow-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h3 className="step-title">Modelo de Deep Learning</h3>
              <p className="step-description">
                CNN multi-escala com mecanismo de atenção analisa a curva de luz.
              </p>
            </div>
          </div>
          <div className="workflow-step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h3 className="step-title">Classificação</h3>
              <p className="step-description">
                O sistema classifica o candidato como Confirmado, Candidato ou Falso Positivo.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="about-section" aria-labelledby="metrics-heading">
        <h2 id="metrics-heading" className="section-heading">Performance do Modelo</h2>
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-value">95%+</div>
            <div className="metric-label">Acurácia</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">&lt;100ms</div>
            <div className="metric-label">Tempo de Predição</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">3 Classes</div>
            <div className="metric-label">Categorias</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">2001</div>
            <div className="metric-label">Pontos de Dados</div>
          </div>
        </div>
      </section>

      <section className="about-section contact-section">
        <h2 className="section-heading">Contato</h2>
        <div className="contact-info">
          <a href="https://github.com" className="contact-link" target="_blank" rel="noopener noreferrer">
            <Github size={20} />
            <span>GitHub</span>
          </a>
          <a href="mailto:contact@example.com" className="contact-link">
            <Mail size={20} />
            <span>Email</span>
          </a>
        </div>
      </section>

      <footer className="about-footer">
        <p>Desenvolvido para o NASA Space Apps Challenge 2025</p>
      </footer>
    </div>
  )
}

export default About
