import React, { useState, useEffect } from 'react';
import { ChevronDown, Calendar } from 'lucide-react';
import './RaceSelector.css';

const RaceSelector = ({ circuits, selectedRace, onRaceSelect }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedYear, setSelectedYear] = useState(2024);
  
  // Available years
  const years = [2022, 2023, 2024, 2025];
  
  // Filter circuits by selected year
  const filteredCircuits = circuits ? circuits.filter(c => c.season === selectedYear) : [];
  
  // Auto-select current year on mount
  useEffect(() => {
    const currentYear = new Date().getFullYear();
    if (years.includes(currentYear)) {
      setSelectedYear(currentYear);
    }
  }, []);

  const handleYearSelect = (year) => {
    setSelectedYear(year);
    // Don't close dropdown, just switch year
  };

  const handleRaceSelect = (race) => {
    onRaceSelect(race);
    setIsOpen(false);
  };

  if (!circuits || circuits.length === 0) {
    return (
      <div className="race-selector">
        <div className="selector-label">SELECT CIRCUIT</div>
        <div className="selector-button disabled">
          <span>LOADING CIRCUITS...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="race-selector">
      <div className="selector-label">SELECT CIRCUIT</div>
      
      <div className="selector-container">
        <button 
          className="selector-button"
          onClick={() => setIsOpen(!isOpen)}
        >
          <div className="selector-content">
            <span className="race-name">
              {selectedRace ? selectedRace.name.toUpperCase() : 'SELECT RACE'}
            </span>
            {selectedRace && (
              <div className="race-meta">
                <span className="race-location">{selectedRace.location}</span>
                <span className="race-divider">|</span>
                <span className="race-round">R{selectedRace.round}</span>
                <span className="race-divider">|</span>
                <span className="race-year">{selectedRace.season}</span>
              </div>
            )}
          </div>
          <ChevronDown 
            size={16} 
            className={`chevron ${isOpen ? 'open' : ''}`}
          />
        </button>

        {isOpen && (
          <div className="selector-dropdown">
            {/* Year Tabs */}
            <div className="year-tabs">
              {years.map(year => (
                <button
                  key={year}
                  className={`year-tab ${selectedYear === year ? 'active' : ''}`}
                  onClick={() => handleYearSelect(year)}
                >
                  <Calendar size={12} />
                  <span>{year}</span>
                </button>
              ))}
            </div>

            {/* Race List for Selected Year */}
            <div className="race-list">
              {filteredCircuits.length > 0 ? (
                filteredCircuits.map((race) => (
                  <button
                    key={`${race.season}-${race.round}`}
                    className={`dropdown-item ${
                      selectedRace?.round === race.round && 
                      selectedRace?.season === race.season ? 'selected' : ''
                    }`}
                    onClick={() => handleRaceSelect(race)}
                  >
                    <div className="dropdown-item-content">
                      <span className="dropdown-race-name">{race.name}</span>
                      <div className="dropdown-race-meta">
                        <span>{race.location}</span>
                        <span className="dropdown-divider">|</span>
                        <span>R{race.round}</span>
                      </div>
                    </div>
                  </button>
                ))
              ) : (
                <div className="no-races">
                  <span>No races available for {selectedYear}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Backdrop */}
      {isOpen && (
        <div 
          className="selector-backdrop"
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  );
};

export default RaceSelector;
