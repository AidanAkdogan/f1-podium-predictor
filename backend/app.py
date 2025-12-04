#!/usr/bin/env python3
"""
Flask API for F1 Race Predictions
Serves predictions to the React frontend
UPDATED: Now supports 2022-2025 seasons
"""

from flask import Flask, jsonify, request
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("⚠️  flask-cors not installed. Install with: pip install flask-cors")
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep import get_clean_laps, get_event_entrants
from src.features import build_features
from src.model import F1PodiumModel
import fastf1

# Initialize Flask app
app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app)  # Allow React to make requests
else:
    # Add manual CORS headers as fallback
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response


# Load model once at startup
model = F1PodiumModel()
try:
    model.load("models/f1_ranker_v1.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️  Could not load model: {e}")
    model = None

# Team colors for frontend
TEAM_COLORS = {
    # 2024 Teams
    'Red Bull Racing': '#0600EF',
    'Ferrari': '#DC0000',
    'Mercedes': '#00D2BE',
    'McLaren': '#FF8700',
    'Aston Martin': '#006F62',
    'Alpine': '#0090FF',
    'Williams': '#005AFF',
    'AlphaTauri': '#2B4562',
    'RB': '#2B4562',
    'Alfa Romeo': '#900000',
    'Haas F1 Team': '#FFFFFF',
    'Kick Sauber': '#52E252',
    
    # Alternative team name formats
    'Red Bull': '#0600EF',
    'Red Bull Racing Honda': '#0600EF',
    'Scuderia Ferrari': '#DC0000',
    'Mercedes-AMG': '#00D2BE',
    'McLaren Mercedes': '#FF8700',
    'Aston Martin Aramco': '#006F62',
    'Alpine Renault': '#0090FF',
    'Williams Mercedes': '#005AFF',
    'Alfa Romeo Racing': '#900000',
    'Haas Ferrari': '#FFFFFF',
    'Sauber': '#52E252',
    'Alfa Romeo Sauber': '#900000',
}

def get_team_color(team_name):
    """
    Get team color with fuzzy matching
    """
    if not team_name or team_name == 'Unknown':
        return '#888888'
    
    # Direct match
    if team_name in TEAM_COLORS:
        return TEAM_COLORS[team_name]
    
    # Fuzzy match - check if any key contains or is contained in team_name
    team_lower = team_name.lower()
    for key, color in TEAM_COLORS.items():
        key_lower = key.lower()
        if key_lower in team_lower or team_lower in key_lower:
            return color
    
    # Check for specific keywords
    if 'red bull' in team_lower:
        return '#0600EF'
    elif 'ferrari' in team_lower:
        return '#DC0000'
    elif 'mercedes' in team_lower:
        return '#00D2BE'
    elif 'mclaren' in team_lower:
        return '#FF8700'
    elif 'aston' in team_lower:
        return '#006F62'
    elif 'alpine' in team_lower:
        return '#0090FF'
    elif 'williams' in team_lower:
        return '#005AFF'
    elif 'alphatauri' in team_lower or team_lower == 'rb':
        return '#2B4562'
    elif 'alfa' in team_lower or 'sauber' in team_lower:
        return '#900000'
    elif 'haas' in team_lower:
        return '#FFFFFF'
    
    # Default gray
    return '#888888'


# Driver photos (placeholder - would need real URLs)
DRIVER_PHOTOS = {
    # Red Bull Racing
    'VER': 'https://www.formula1.com/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png.transform/1col/image.png',
    'PER': 'https://www.formula1.com/content/dam/fom-website/drivers/S/SERPER01_Sergio_Perez/serper01.png.transform/1col/image.png',
    
    # Ferrari
    'LEC': 'https://www.formula1.com/content/dam/fom-website/drivers/C/CHARLEC01_Charles_Leclerc/charlec01.png.transform/1col/image.png',
    'SAI': 'https://www.formula1.com/content/dam/fom-website/drivers/C/CARSAI01_Carlos_Sainz/carsai01.png.transform/1col/image.png',
    
    # Mercedes
    'HAM': 'https://www.formula1.com/content/dam/fom-website/drivers/L/LEWHAM01_Lewis_Hamilton/lewham01.png.transform/1col/image.png',
    'RUS': 'https://www.formula1.com/content/dam/fom-website/drivers/G/GEORUS01_George_Russell/georus01.png.transform/1col/image.png',
    
    # McLaren
    'NOR': 'https://www.formula1.com/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png.transform/1col/image.png',
    'PIA': 'https://www.formula1.com/content/dam/fom-website/drivers/O/OSCPIA01_Oscar_Piastri/oscpia01.png.transform/1col/image.png',
    
    # Aston Martin
    'ALO': 'https://www.formula1.com/content/dam/fom-website/drivers/F/FERALO01_Fernando_Alonso/feralo01.png.transform/1col/image.png',
    'STR': 'https://www.formula1.com/content/dam/fom-website/drivers/L/LANSTR01_Lance_Stroll/lanstr01.png.transform/1col/image.png',
    
    # Alpine
    'GAS': 'https://www.formula1.com/content/dam/fom-website/drivers/P/PIEGAS01_Pierre_Gasly/piegas01.png.transform/1col/image.png',
    'OCO': 'https://www.formula1.com/content/dam/fom-website/drivers/E/ESTOCO01_Esteban_Ocon/estoco01.png.transform/1col/image.png',
    
    # Williams
    'ALB': 'https://www.formula1.com/content/dam/fom-website/drivers/A/ALEALB01_Alexander_Albon/alealb01.png.transform/1col/image.png',
    'SAR': 'https://www.formula1.com/content/dam/fom-website/drivers/L/LOGSAR01_Logan_Sargeant/logsar01.png.transform/1col/image.png',
    'COL': 'https://www.formula1.com/content/dam/fom-website/drivers/F/FRANCO01_Franco_Colapinto/fracol01.png.transform/1col/image.png',
    
    # RB / AlphaTauri
    'TSU': 'https://www.formula1.com/content/dam/fom-website/drivers/Y/YUKTSU01_Yuki_Tsunoda/yuktsu01.png.transform/1col/image.png',
    'RIC': 'https://www.formula1.com/content/dam/fom-website/drivers/D/DANRIC01_Daniel_Ricciardo/danric01.png.transform/1col/image.png',
    'LAW': 'https://www.formula1.com/content/dam/fom-website/drivers/L/LIALAW01_Liam_Lawson/lialaw01.png.transform/1col/image.png',
    
    # Kick Sauber / Alfa Romeo
    'BOT': 'https://www.formula1.com/content/dam/fom-website/drivers/V/VALBOT01_Valtteri_Bottas/valbot01.png.transform/1col/image.png',
    'ZHO': 'https://www.formula1.com/content/dam/fom-website/drivers/G/GUAZHO01_Guanyu_Zhou/guazho01.png.transform/1col/image.png',
    
    # Haas
    'HUL': 'https://www.formula1.com/content/dam/fom-website/drivers/N/NICHUL01_Nico_Hulkenberg/nichul01.png.transform/1col/image.png',
    'MAG': 'https://www.formula1.com/content/dam/fom-website/drivers/K/KEVMAG01_Kevin_Magnussen/kevmag01.png.transform/1col/image.png',
    'BEA': 'https://www.formula1.com/content/dam/fom-website/drivers/O/OLIBEA01_Oliver_Bearman/olibea01.png.transform/1col/image.png',
}


@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/api/circuits', methods=['GET'])
def get_circuits():
    """
    Get list of available circuits/races from 2022-2025
    
    Query params:
        season (optional): Filter by specific season
    """
    
    season_filter = request.args.get('season', type=int)
    
    try:
        all_circuits = []
        
        # Define years to fetch
        years = [season_filter] if season_filter else [2022, 2023, 2024, 2025]
        
        for year in years:
            try:
                schedule = fastf1.get_event_schedule(year)
                
                for _, event in schedule.iterrows():
                    all_circuits.append({
                        'round': int(event['RoundNumber']),
                        'name': event['EventName'],
                        'location': event['Location'],
                        'date': str(event['EventDate']),
                        'country': event.get('Country', 'Unknown'),
                        'season': year  # IMPORTANT: Include season
                    })
            
            except Exception as e:
                # It's OK if 2025 doesn't have full data yet
                print(f"⚠️  Could not fetch {year} schedule: {e}")
                continue
        
        if not all_circuits:
            return jsonify({'error': 'No circuits found'}), 404
        
        return jsonify(all_circuits)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_race():
    """
    Predict podium for a specific race
    
    Request body:
    {
        "season": 2024,  (required - now supports 2022-2025)
        "round": 20      (required)
    }
    """
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        season = data.get('season')
        round_no = data.get('round')
        
        # Validate inputs
        if not season:
            return jsonify({'error': 'Season number required'}), 400
        if not round_no:
            return jsonify({'error': 'Round number required'}), 400
        
        # Validate season range
        if season not in [2022, 2023, 2024, 2025]:
            return jsonify({'error': 'Season must be between 2022-2025'}), 400
        
        # Load session data
        try:
            practice, quali, race, event, group_size = get_clean_laps(season, round_no)
        except Exception as e:
            error_msg = str(e).lower()
            # Check if it's a "race hasn't happened yet" error
            if any(term in error_msg for term in ['no data', 'not found', 'no session', 'cannot load', 'error loading']):
                return jsonify({
                    'error': 'Race data not yet available',
                    'message': f'This race has not been completed yet. Check back after the race weekend!',
                    'details': str(e)
                }), 404
            raise
        
        if practice.empty and quali.empty:
            return jsonify({
                'error': 'Race data not yet available',
                'message': 'No practice or qualifying data available for this race.'
            }), 404
        
        # Build features
        feats = build_features(
            practice, quali, pd.DataFrame(), event,
            is_training=False,
            race_completed=False
        )
        
        if feats.empty:
            return jsonify({'error': 'Could not build features'}), 500
        
        # Make prediction
        X_preprocessed = model.preprocessor.transform(feats)
        scores = model.model.predict(X_preprocessed)
        
        # Create results
        results = []
        for idx, row in feats.iterrows():
            driver = row['Driver']
            team = row.get('team', 'Unknown')
            
            # Use improved color lookup
            team_color = get_team_color(team)
            
            results.append({
                'position': None,  # Will be set after sorting
                'driver': driver,
                'team': team,
                'score': float(scores[idx]),
                'teamColor': team_color,
                'photo': DRIVER_PHOTOS.get(driver, None),
                # Feature contributions (simplified)
                'features': {
                    'quali_rank': float(row.get('quali_rank', 0)) if pd.notna(row.get('quali_rank', 0)) else 0.0,
                    'form_rating': float(row.get('form_podium_rate_l5', 0)) if pd.notna(row.get('form_podium_rate_l5', 0)) else 0.0,
                    'race_pace': float(1 / (row.get('race_pace_fp2_best', 100) + 0.1)) if pd.notna(row.get('race_pace_fp2_best')) else 0.0,
                }
            })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Add positions
        for i, result in enumerate(results, 1):
            result['position'] = i
        
        # Extract podium with podium_step
        podium = results[:3]
        for i, p in enumerate(podium):
            p['podium_step'] = i + 1  # 1, 2, 3
        
        # Get event info - handle both dict and pandas Series
        if event is not None:
            if isinstance(event, dict):
                event_name = event.get('EventName', 'Unknown')
                event_location = event.get('Location', 'Unknown')
            else:
                # pandas Series
                event_name = str(event.get('EventName', 'Unknown')) if hasattr(event, 'get') else 'Unknown'
                event_location = str(event.get('Location', 'Unknown')) if hasattr(event, 'get') else 'Unknown'
        else:
            event_name = 'Unknown'
            event_location = 'Unknown'
        
        event_info = {
            'name': event_name,
            'location': event_location,
            'season': int(season),
            'round': int(round_no)
        }
        
        # Get actual results if race completed
        actual_results = []
        if not race.empty:
            try:
                race_results = race.session.results
                if race_results is not None:
                    actual_top10 = race_results.nsmallest(10, 'Position')
                    for _, row in actual_top10.iterrows():
                        actual_results.append({
                            'position': int(row['Position']),
                            'driver': str(row['Abbreviation']),
                            'team': str(row.get('TeamName', 'Unknown'))
                        })
            except Exception as e:
                print(f"Could not get actual results: {e}")
                pass
        
        # IMPROVED MODEL CONFIDENCE CALCULATION
        # Instead of simple average, calculate based on score separation
        all_scores = [r['score'] for r in results]
        
        if len(all_scores) >= 10:
            # Method 1: Top 3 separation from rest
            top3_avg = np.mean(all_scores[:3])
            rest_avg = np.mean(all_scores[3:10])
            
            # Method 2: Score variance (lower = more confident)
            top3_variance = np.var(all_scores[:3])
            
            # Method 3: Gap between P3 and P4
            p3_p4_gap = all_scores[2] - all_scores[3] if len(all_scores) >= 4 else 0
            
            # Normalize to 0-1 range
            max_score = max(all_scores)
            min_score = min(all_scores)
            score_range = max_score - min_score if max_score != min_score else 1.0
            
            # Calculate confidence components
            separation_conf = (top3_avg - rest_avg) / score_range if score_range > 0 else 0.5
            variance_conf = 1 - min(1.0, top3_variance / 0.5)  # Lower variance = higher confidence
            gap_conf = min(1.0, p3_p4_gap / 0.3) if score_range > 0 else 0.5
            
            # Weighted combination (50% separation, 30% variance, 20% gap)
            confidence = 0.5 * separation_conf + 0.3 * variance_conf + 0.2 * gap_conf
            
            # Ensure confidence is in reasonable range (40% - 95%)
            confidence = max(0.40, min(0.95, confidence))
            
        else:
            confidence = 0.70  # Default for limited data
        
        response = {
            'event': event_info,
            'predictions': results,
            'actual': actual_results if actual_results else None,
            'podium': podium,
            'modelConfidence': float(confidence)
        }
        
        print(f"✅ Prediction for {season} R{round_no} - Confidence: {confidence:.2%}")
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Check if it's a data availability error
        error_msg = str(e).lower()
        if any(term in error_msg for term in ['no data', 'not found', 'no session', 'cannot load']):
            return jsonify({
                'error': 'Race data not yet available',
                'message': 'This race has not been completed yet. Check back after the race weekend!'
            }), 404
        
        return jsonify({'error': str(e)}), 500


@app.route('/api/next-race', methods=['GET'])
def next_race():
    """Get the next upcoming race (checks 2024 and 2025)"""
    
    try:
        from datetime import datetime
        
        now = datetime.now()
        current_year = now.year
        
        # Check current year and next year
        for year in [current_year, current_year + 1]:
            if year > 2025:  # Don't go beyond 2025
                break
                
            try:
                schedule = fastf1.get_event_schedule(year)
                
                # Find next race
                for _, event in schedule.iterrows():
                    event_date = pd.to_datetime(event['EventDate'])
                    if event_date > now:
                        return jsonify({
                            'season': year,
                            'round': int(event['RoundNumber']),
                            'name': event['EventName'],
                            'location': event['Location'],
                            'date': str(event['EventDate'])
                        })
            except Exception as e:
                print(f"⚠️  Could not fetch {year} schedule: {e}")
                continue
        
        # If no upcoming races found, return last race of 2024
        schedule = fastf1.get_event_schedule(2024)
        last_event = schedule.iloc[-1]
        return jsonify({
            'season': 2024,
            'round': int(last_event['RoundNumber']),
            'name': last_event['EventName'],
            'location': last_event['Location'],
            'date': str(last_event['EventDate'])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n🏎️  F1 Race Predictor API (2022-2025)")
    print("=" * 50)
    print("Starting server on http://localhost:5000")
    print("API endpoints:")
    print("  GET  /api/health       - Health check")
    print("  GET  /api/circuits     - List all circuits (2022-2025)")
    print("  GET  /api/circuits?season=2024 - Filter by season")
    print("  GET  /api/next-race    - Get next race info")
    print("  POST /api/predict      - Predict race results")
    print("=" * 50)
    print("\n📅 Supported seasons: 2022, 2023, 2024, 2025")
    print("⚠️  Future 2025 races will return friendly errors until completed\n")
    
    app.run(debug=True, port=5000)
