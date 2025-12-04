import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './FeatureInsights.css';

const FeatureInsights = ({ predictions }) => {
  // Prepare data for visualization
  const qualiData = predictions.map(p => ({
    driver: p.driver,
    value: p.features.quali_rank || 20,
    color: p.teamColor
  })).sort((a, b) => a.value - b.value);

  const formData = predictions.map(p => ({
    driver: p.driver,
    value: (p.features.form_rating || 0) * 100,
    color: p.teamColor
  })).sort((a, b) => b.value - a.value);

  const paceData = predictions.map(p => ({
    driver: p.driver,
    value: p.features.race_pace || 0,
    color: p.teamColor
  })).sort((a, b) => b.value - a.value);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-driver">{payload[0].payload.driver}</p>
          <p className="tooltip-value">{payload[0].value.toFixed(2)}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="feature-insights">
      <div className="insight-grid">
        {/* Qualifying Performance */}
        <div className="insight-card">
          <h3 className="insight-title">Qualifying Performance</h3>
          <p className="insight-subtitle">Grid positions (lower is better)</p>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={qualiData} margin={{ top: 10, right: 10, left: 0, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis 
                dataKey="driver" 
                angle={-45} 
                textAnchor="end" 
                height={80}
                stroke="#999"
                tick={{ fill: '#999', fontSize: 11 }}
              />
              <YAxis 
                reversed 
                stroke="#999"
                tick={{ fill: '#999' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {qualiData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color || '#888'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Form Rating */}
        <div className="insight-card">
          <h3 className="insight-title">Recent Form</h3>
          <p className="insight-subtitle">Last 5 races podium rate (%)</p>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={formData} margin={{ top: 10, right: 10, left: 0, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis 
                dataKey="driver" 
                angle={-45} 
                textAnchor="end" 
                height={80}
                stroke="#999"
                tick={{ fill: '#999', fontSize: 11 }}
              />
              <YAxis 
                stroke="#999"
                tick={{ fill: '#999' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {formData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color || '#888'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Race Pace */}
        <div className="insight-card">
          <h3 className="insight-title">Race Pace Indicator</h3>
          <p className="insight-subtitle">FP2 pace relative strength</p>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={paceData} margin={{ top: 10, right: 10, left: 0, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis 
                dataKey="driver" 
                angle={-45} 
                textAnchor="end" 
                height={80}
                stroke="#999"
                tick={{ fill: '#999', fontSize: 11 }}
              />
              <YAxis 
                stroke="#999"
                tick={{ fill: '#999' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {paceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color || '#888'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="insights-footer">
        <p className="insights-note">
          📊 These metrics are derived from practice sessions, qualifying, driver form, 
          and historical track performance. The model considers 30+ features to make predictions.
        </p>
      </div>
    </div>
  );
};

export default FeatureInsights;
