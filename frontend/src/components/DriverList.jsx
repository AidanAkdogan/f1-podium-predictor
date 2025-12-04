import React from 'react';
import { TrendingUp, TrendingDown, Check, X, Minus } from 'lucide-react';
import './DriverList.css';

const DriverList = ({ predictions, actualResults }) => {
  const getActualPosition = (driver) => {
    if (!actualResults) return null;
    const actual = actualResults.find(r => r.driver === driver);
    return actual?.position;
  };

  const getDelta = (predicted, actual) => {
    if (!actual) return null;
    return predicted - actual;
  };

  const getMedalEmoji = (position) => {
    switch (position) {
      case 1: return '🏆';
      case 2: return '🥈';
      case 3: return '🥉';
      default: return null;
    }
  };

  const getAccuracyIcon = (predicted, actual) => {
    if (!actual) return null;
    const diff = Math.abs(predicted - actual);
    if (diff === 0) return <Check size={16} className="accuracy-perfect" />;
    if (diff <= 2) return <Check size={16} className="accuracy-close" />;
    if (diff <= 5) return <Minus size={16} className="accuracy-ok" />;
    return <X size={16} className="accuracy-poor" />;
  };

  return (
    <div className="race-classification">
      {/* Header */}
      <div className="classification-header">
        {/* ADDED: Empty spacer to match the row's colored border */}
        <div className="header-spacer"></div> 
        <div className="col-position-header">POS</div>
        <div className="col-driver-header">DRIVER</div>
        <div className="col-team-header">TEAM</div>
        <div className="col-score-header">SCORE</div>
        <div className="col-details-header">DETAILS</div>
        {actualResults && <div className="col-actual-header">ACTUAL</div>}
        {actualResults && <div className="col-delta-header">DELTA</div>}
      </div>

      {/* Rows */}
      <div className="classification-body">
        {predictions.map((driver) => {
          const actualPos = getActualPosition(driver.driver);
          const delta = getDelta(driver.position, actualPos);
          const isPodium = driver.position <= 3;
          const medalEmoji = getMedalEmoji(driver.position);

          return (
            <div 
              key={driver.driver}
              className={`classification-row ${isPodium ? `podium-row podium-${driver.position}` : ''}`}
            >
              {/* Team Color Border */}
              <div 
                className="team-border"
                style={{ backgroundColor: driver.teamColor || '#888888' }}
              />

              {/* Position */}
              <div className="col-position-data">
                {medalEmoji && <span className="medal-icon">{medalEmoji}</span>}
                <span className="position-number">{driver.position}</span>
              </div>

              {/* Driver */}
              <div className="col-driver-data">
                <span className="driver-code-text">{driver.driver}</span>
              </div>

              {/* Team */}
              <div className="col-team-data">
                <span className="team-name-text">{driver.team}</span>
              </div>

              {/* Score */}
              <div className="col-score-data">
                <div className="score-with-dots">
                  <div className="score-dots">
                    {[1, 2, 3, 4, 5].map((dot) => (
                      <div
                        key={dot}
                        className={`score-dot ${
                          dot <= Math.round((driver.score / 1.2) * 5) ? 'filled' : ''
                        }`}
                        style={{
                          backgroundColor: dot <= Math.round((driver.score / 1.2) * 5)
                            ? driver.teamColor || '#888888'
                            : 'transparent'
                        }}
                      />
                    ))}
                  </div>
                  <span className="score-number-text">{driver.score.toFixed(3)}</span>
                </div>
              </div>

              {/* Details */}
              <div className="col-details-data">
                {driver.features?.quali_rank && (
                  <span className="detail-chip quali-chip">
                    Q:P{Math.round(driver.features.quali_rank)}
                  </span>
                )}
                {driver.features?.form_rating > 0 && (
                  <span className="detail-chip form-chip">
                    F:{(driver.features.form_rating * 100).toFixed(0)}%
                  </span>
                )}
              </div>

              {/* Actual */}
              {actualResults && (
                <div className="col-actual-data">
                  {actualPos ? (
                    <div className="actual-with-icon">
                      {getAccuracyIcon(driver.position, actualPos)}
                      <span className="actual-position-text">P{actualPos}</span>
                    </div>
                  ) : (
                    <span className="data-empty-text">-</span>
                  )}
                </div>
              )}

              {/* Delta */}
              {actualResults && (
                <div className="col-delta-data">
                  {delta !== null && delta !== 0 ? (
                    <div className={`delta-badge ${delta > 0 ? 'delta-better' : 'delta-worse'}`}>
                      {delta > 0 ? (
                        <TrendingUp size={14} strokeWidth={2.5} />
                      ) : (
                        <TrendingDown size={14} strokeWidth={2.5} />
                      )}
                      <span>{Math.abs(delta)}</span>
                    </div>
                  ) : delta === 0 ? (
                    <div className="delta-badge delta-perfect">
                      <Check size={14} strokeWidth={2.5} />
                    </div>
                  ) : (
                    <span className="data-empty-text">-</span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default DriverList;
