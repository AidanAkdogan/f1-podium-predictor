import React from 'react';
import { Trophy, Award } from 'lucide-react';
import './PodiumDisplay.css';

const PodiumDisplay = ({ podium }) => {
  if (!podium || podium.length < 3) {
    return (
      <div className="podium-empty">
        <Trophy size={32} />
        <span>No podium data available</span>
      </div>
    );
  }

  const getMedalEmoji = (position) => {
    switch (position) {
      case 1: return '🏆';
      case 2: return '🥈';
      case 3: return '🥉';
      default: return '';
    }
  };

  const getMedalGradient = (position) => {
    switch (position) {
      case 1: return 'linear-gradient(135deg, #ffb800, #ffd700)';
      case 2: return 'linear-gradient(135deg, #c0c0c0, #e8e8e8)';
      case 3: return 'linear-gradient(135deg, #cd7f32, #e89b5a)';
      default: return 'transparent';
    }
  };

  // Visual order: P2, P1, P3
  const visualOrder = [podium[1], podium[0], podium[2]];

  return (
    <div className="victory-cards">
      {visualOrder.map((driver, idx) => {
        const position = driver.podium_step || driver.position;
        const teamColor = driver.teamColor || '#888888';
        const isWinner = position === 1;
        
        return (
          <div 
            key={driver.driver}
            className={`victory-card position-${position} ${isWinner ? 'winner' : ''}`}
            style={{
              '--medal-gradient': getMedalGradient(position)
            }}
          >
            {/* Medal Badge */}
            <div className="medal-badge">
              <span className="medal-emoji">{getMedalEmoji(position)}</span>
              <span className="position-text">P{position}</span>
            </div>

            {/* Driver Photo */}
            <div className="driver-photo-box">
              {driver.photo ? (
                <img src={driver.photo} alt={driver.driver} className="driver-photo" />
              ) : (
                <div className="driver-photo-placeholder">
                  <span>{driver.driver}</span>
                </div>
              )}
            </div>

            {/* Driver Info */}
            <div className="driver-info-box">
              <div className="driver-code-large">{driver.driver}</div>
              <div className="driver-team-text">{driver.team}</div>
            </div>

            {/* Team Color Accent */}
            <div 
              className="team-color-accent"
              style={{ backgroundColor: teamColor }}
            />

            {/* Score */}
            <div className="score-display-box">
              <div className="score-label-text">SCORE</div>
              <div className="score-number-large">{driver.score.toFixed(3)}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default PodiumDisplay;
