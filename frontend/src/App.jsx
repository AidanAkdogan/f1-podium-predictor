import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Trophy, Calendar, MapPin, TrendingUp, Loader2, AlertCircle, X, RefreshCw } from 'lucide-react';
import PodiumDisplay from './components/PodiumDisplay';
import DriverList from './components/DriverList';
import RaceSelector from './components/RaceSelector';
import FeatureInsights from './components/FeatureInsights';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function App() {
  const [circuits, setCircuits] = useState([]);
  const [selectedRace, setSelectedRace] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState(false);

  // Check API health on mount
  useEffect(() => {
    axios.get(`${API_BASE}/health`)
      .then(res => {
        setApiHealth(res.data.model_loaded);
        if (!res.data.model_loaded) {
          setError('Model not loaded. Please train the model first.');
        }
      })
      .catch(err => {
        setError('Cannot connect to backend API. Make sure Flask is running.');
        console.error(err);
      });
  }, []);

  // Load circuits
  useEffect(() => {
    if (apiHealth) {
      axios.get(`${API_BASE}/circuits`)
        .then(res => setCircuits(res.data))
        .catch(err => {
          console.error('Failed to load circuits:', err);
          setError('Failed to load race list');
        });
    }
  }, [apiHealth]);

  // Load next race on mount
  useEffect(() => {
    if (apiHealth) {
      axios.get(`${API_BASE}/next-race`)
        .then(res => {
          setSelectedRace(res.data);
          handlePredict(res.data);
        })
        .catch(err => console.error('Failed to load next race:', err));
    }
  }, [apiHealth]);

  const handlePredict = async (race) => {
    if (!race) return;
    
    setLoading(true);
    setError(null);
    setPredictions(null);

    try {
      const response = await axios.post(`${API_BASE}/predict`, {
        season: race.season,
        round: race.round
      });
      
      const data = response.data;
      
      // Validate response
      if (!data || !data.predictions || !Array.isArray(data.predictions)) {
        throw new Error('Invalid response format from server');
      }
      
      // Ensure all required fields exist
      if (!data.event) {
        data.event = {
          name: race.name || 'Unknown Race',
          location: race.location || 'Unknown',
          season: race.season,
          round: race.round
        };
      }
      
      if (!data.podium || data.podium.length < 3) {
        data.podium = data.predictions.slice(0, 3).map((p, idx) => ({
          ...p,
          podium_step: idx + 1
        }));
      }
      
      if (typeof data.modelConfidence !== 'number') {
        data.modelConfidence = 0.75;
      }
      
      setPredictions(data);
    } catch (err) {
      const errorMsg = err.response?.data?.error || err.message || 'Failed to get predictions';
      setError(errorMsg);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRaceSelect = (race) => {
    setSelectedRace(race);
    handlePredict(race);
  };

  const handleRetry = () => {
    if (selectedRace) {
      handlePredict(selectedRace);
    }
  };

  const dismissError = () => {
    setError(null);
  };

  if (!apiHealth) {
    return (
      <div className="app error-screen">
        <div className="error-container">
          <AlertCircle size={64} className="error-icon" />
          <h1>Backend Connection Error</h1>
          <p>{error || 'Connecting to backend...'}</p>
          <div className="error-help">
            <h3>Setup Checklist:</h3>
            <ol>
              <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
              <li>Train model: <code>python scripts/train_all_races.py --seasons 2022 2023 2024</code></li>
              <li>Start Flask: <code>python backend/app.py</code></li>
              <li>Refresh this page</li>
            </ol>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <Trophy className="logo-icon" size={40} />
            <div>
              <h1>F1 Podium Predictor</h1>
              <p className="tagline">AI-Powered Race Predictions</p>
            </div>
          </div>
          <div className="header-stats">
            <div className="stat">
              <MapPin size={16} />
              <span>{circuits.length} Circuits</span>
            </div>
            <div className="stat">
              <TrendingUp size={16} />
              <span>XGBoost Ranker</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Race Selector */}
        <section className="race-selector-section">
          <RaceSelector
            circuits={circuits}
            selectedRace={selectedRace}
            onRaceSelect={handleRaceSelect}
          />
        </section>

        {/* Error Message - Dismissible */}
        {error && !loading && (
          <div className="error-banner">
            <div className="error-banner-content">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
            <div className="error-banner-actions">
              <button onClick={handleRetry} className="retry-button">
                <RefreshCw size={16} />
                Retry
              </button>
              <button onClick={dismissError} className="dismiss-button">
                <X size={16} />
              </button>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="loading-container">
            <Loader2 className="spinner" size={48} />
            <p>Analyzing race data for {selectedRace?.name || 'selected race'}...</p>
          </div>
        )}

        {/* Predictions */}
        {predictions && !loading && predictions.event && (
          <>
            {/* Event Info */}
            <section className="event-info">
              <div className="event-details">
                <h2>{predictions.event.name}</h2>
                <div className="event-meta">
                  <span><MapPin size={16} /> {predictions.event.location}</span>
                  <span><Calendar size={16} /> Round {predictions.event.round}</span>
                </div>
              </div>
              <div className="model-confidence">
                <div className="confidence-label">Model Confidence</div>
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill" 
                    style={{ width: `${(predictions.modelConfidence || 0.75) * 100}%` }}
                  />
                </div>
                <div className="confidence-value">
                  {((predictions.modelConfidence || 0.75) * 100).toFixed(1)}%
                </div>
              </div>
            </section>

            {/* Podium Display */}
            {predictions.podium && predictions.podium.length >= 3 && (
              <section className="podium-section">
                <h2 className="section-title">
                  <Trophy size={24} />
                  Predicted Podium
                </h2>
                <PodiumDisplay podium={predictions.podium} />
              </section>
            )}

            {/* Full Grid */}
            {predictions.predictions && predictions.predictions.length > 0 && (
              <section className="grid-section">
                <h2 className="section-title">Full Race Prediction</h2>
                <DriverList 
                  predictions={predictions.predictions}
                  actualResults={predictions.actual}
                />
              </section>
            )}

            {/* Feature Insights */}
            {predictions.predictions && predictions.predictions.length >= 10 && (
              <section className="insights-section">
                <h2 className="section-title">
                  <TrendingUp size={24} />
                  Key Performance Indicators
                </h2>
                <FeatureInsights predictions={predictions.predictions.slice(0, 10)} />
              </section>
            )}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>Powered by FastF1, XGBoost & React | Data from 2022-2024 seasons</p>
        <p className="footer-note">
          Prediction accuracy indicator: ✅ Perfect/Close (≤2) | ⚠️ OK (≤5) | ❌ Poor (&gt;5)
        </p>
      </footer>
    </div>
  );
}

export default App;
