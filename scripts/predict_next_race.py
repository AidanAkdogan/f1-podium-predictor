"""
Predicts a podium for upcoming/past races.
"""

import sys
from pathlib import Path
import pandas as pd
import fastf1

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep import get_clean_laps, CACHE_DIR
from src.features import build_features
from src.model import F1PodiumModel

fastf1.Cache.enable_cache(str(CACHE_DIR))


def predict_race(season: int, round_no: int, show_actual: bool = True):
    """
    Predict podium for a specific race.
    
    Args:
        season: Year (e.g., 2024)
        round_no: Round number (e.g., 20)
        show_actual: If True, show actual results for comparison (if race completed)
    """
    print(f"\n{'='*60}")
    print(f"🏎️  F1 PODIUM PREDICTION")
    print(f"{'='*60}")
    print(f"Season: {season}")
    print(f"Round: {round_no}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading session data...")
    practice, quali, race, event, group_size = get_clean_laps(season, round_no)
    
    # Check if event loaded properly
    if event is None or (hasattr(event, 'empty') and event.empty):
        print("⚠️  Event metadata unavailable")
        event_name = "Unknown"
        location = "Unknown"
    else:
        event_name = event.get('EventName', event.get('OfficialEventName', 'Unknown'))
        location = event.get('Location', 'Unknown')
        print(f"📍 Event: {event_name}")
        print(f"📍 Location: {location}\n")
    
    # Check data availability
    if practice.empty and quali.empty:
        print("❌ No practice or qualifying data available")
        return
    
    print(f"✅ Practice laps: {len(practice)}")
    print(f"✅ Quali laps: {len(quali)}")
    print(f"✅ Entrants: {group_size}\n")
    
    # Build features
    print("Building features...")
    race_completed = not race.empty
    feats = build_features(
        practice, quali, race, event,
        is_training=False,
        race_completed=False  # Don't use race results for prediction
    )
    
    if feats.empty:
        print("❌ Failed to build features")
        return
    
    print(f"✅ Features built: {feats.shape}\n")
    
    # Load model and predict
    print("Loading model...")
    model = F1PodiumModel()
    model.load("models/f1_ranker_v1.pkl")
    
    print("Predicting podium...\n")
    podium = model.predict_podium(feats)
    
    # Get full predictions for all drivers
    X_preprocessed = model.preprocessor.transform(feats)
    all_scores = model.model.predict(X_preprocessed)
    
    # Add scores to the features dataframe for full ranking
    feats_with_scores = feats.copy()
    feats_with_scores['predicted_score'] = all_scores
    feats_with_scores['predicted_position'] = feats_with_scores['predicted_score'].rank(method='first', ascending=False).astype(int)
    
    # Display prediction
    print(f"{'='*60}")
    print(f"🏆 PREDICTED PODIUM")
    print(f"{'='*60}")
    
    for idx, row in podium.iterrows():
        position_emoji = ["🥇", "🥈", "🥉"][idx]
        driver = row['Driver']
        # Get score from full predictions
        driver_score = feats_with_scores[feats_with_scores['Driver'] == driver]['predicted_score'].values[0]
        print(f"{position_emoji} P{int(row['podium_step'])}: {driver} (score: {driver_score:.3f})")
    
    # Show all predictions
    print(f"\n{'='*60}")
    print(f"📊 FULL PREDICTION (Top 10)")
    print(f"{'='*60}")
    
    full_pred = feats_with_scores[['Driver', 'team', 'predicted_score', 'predicted_position']].sort_values('predicted_position').head(10)
    
    for idx, row in full_pred.iterrows():
        print(f"P{row['predicted_position']:2d}: {row['Driver']:3s} ({row['team']})")
    
    # Compare with actual if race completed
    if show_actual and race_completed:
        print(f"\n{'='*60}")
        print(f"🏁 ACTUAL RESULTS (for comparison)")
        print(f"{'='*60}")
        
        try:
            session = fastf1.get_session(season, round_no, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            results = session.results
            
            if results is not None and not results.empty:
                actual_top10 = results.nsmallest(10, 'Position')[['Abbreviation', 'Position', 'TeamName']]
                
                for _, row in actual_top10.iterrows():
                    marker = "✅" if int(row['Position']) <= 3 else "  "
                    print(f"{marker} P{int(row['Position']):2d}: {row['Abbreviation']:3s} ({row['TeamName']})")
                
                # Calculate accuracy
                actual_podium = set(results.nsmallest(3, 'Position')['Abbreviation'].values)
                predicted_podium = set(podium['Driver'].values)
                correct = len(actual_podium & predicted_podium)
                
                print(f"\n{'='*60}")
                print(f"📊 PREDICTION ACCURACY")
                print(f"{'='*60}")
                print(f"Podium positions correct: {correct}/3 ({100*correct/3:.0f}%)")
                
                if correct == 3:
                    print("🎉 Perfect prediction!")
                elif correct >= 2:
                    print("✅ Good prediction!")
                elif correct >= 1:
                    print("⚠️  Fair prediction")
                else:
                    print("❌ Missed the podium")
                
        except Exception as e:
            print(f"⚠️  Could not load actual results: {e}")
    
    print(f"\n{'='*60}\n")


def predict_upcoming_races(season: int = 2024):
    """Find and predict upcoming races."""
    print(f"🔍 Looking for upcoming races in {season}...\n")
    
    schedule = fastf1.get_event_schedule(season)
    
    # Find races that haven't happened yet or recent ones
    from datetime import datetime, timedelta
    now = datetime.now()
    recent_cutoff = now - timedelta(days=7)
    
    relevant_races = []
    
    for _, event in schedule.iterrows():
        event_date = pd.to_datetime(event['EventDate'])
        
        # Include upcoming races and races from last 7 days
        if event_date >= recent_cutoff:
            relevant_races.append({
                'round': int(event['RoundNumber']),
                'name': event['EventName'],
                'date': event_date,
                'status': 'upcoming' if event_date > now else 'recent'
            })
    
    if not relevant_races:
        print("No upcoming races found. Showing last 3 races instead.\n")
        for i in range(max(1, len(schedule) - 2), len(schedule) + 1):
            try:
                event = schedule[schedule['RoundNumber'] == i].iloc[0]
                relevant_races.append({
                    'round': i,
                    'name': event['EventName'],
                    'date': pd.to_datetime(event['EventDate']),
                    'status': 'past'
                })
            except:
                pass
    
    # Show options
    print("Available races:")
    for i, race in enumerate(relevant_races, 1):
        status_icon = "🔜" if race['status'] == 'upcoming' else "🏁"
        print(f"{i}. {status_icon} R{race['round']}: {race['name']} ({race['date'].strftime('%Y-%m-%d')})")
    
    print(f"\nTo predict a race:")
    print(f"  python scripts/predict_next_race.py --season {season} --round <round_number>")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict F1 race podium")
    parser.add_argument('--season', type=int, default=2024, help='Season year')
    parser.add_argument('--round', type=int, help='Round number')
    parser.add_argument('--no-actual', action='store_true', help='Hide actual results')
    parser.add_argument('--list', action='store_true', help='List upcoming races')
    
    args = parser.parse_args()
    
    if args.list or args.round is None:
        predict_upcoming_races(args.season)
        return
    
    predict_race(args.season, args.round, show_actual=not args.no_actual)


if __name__ == '__main__':
    main()