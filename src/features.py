import pandas as pd
import numpy as np
from src.data_prep import  get_clean_laps, get_driver_form, get_driver_track_history, get_event_entrants, load_session, get_track_conditions
from typing import Optional, Tuple
from sklearn.linear_model import LinearRegression

# NEW: Global imputation defaults (computed from historical data or approximated; update with real vals later)
GLOBAL_PODIUM_RATE = 0.15  # Approx 3/20 drivers podium per race
GLOBAL_DNF_RATE = 0.10     # Approx 2/20 DNF per race

def build_features(practice: pd.DataFrame, quali: pd.DataFrame, race: pd.DataFrame, event: pd.DataFrame, is_training: bool = False, race_completed: bool = False) -> pd.DataFrame:  # Added race_completed flag
    if practice.empty and quali.empty:
        return pd.DataFrame()

    is_sprint_weekend = False
    if not practice.empty:
        sessions_available = practice["Session"].unique()
        is_sprint_weekend = len(sessions_available) == 1 and "FP1" in sessions_available
    
    # Adapt data sources based on weekend format
    if is_sprint_weekend:
        # Sprint: only FP1 available, treat it as "FP2" for features
        fp2 = practice.copy()
        fp3 = pd.DataFrame()
        print(f"  🏃 Sprint weekend detected - using FP1 as baseline")
    else:
        # Normal: separate FP2 and FP3
        fp2 = practice[practice["Session"] == "FP2"].copy()
        fp3 = practice[practice["Session"] == "FP3"].copy()

    # NEW: Filter drivers to entrants early (saves compute, as per feedback)
    season = event.get("EventDate", pd.NaT).year if event.get("EventDate") else 2024
    round_no = event.get("RoundNumber", 0)
    entrants = get_event_entrants(season, round_no)
    drivers = pd.Index(
        [d for d in pd.concat([practice, quali], ignore_index=True)["Driver"].unique() if d in entrants]
    ).unique()
    
    fp2 = practice[practice["Session"] == "FP2"].copy()
    fp3 = practice[practice["Session"] == "FP3"].copy()

    location = event.get("Location", "Unknown")


    # === TRACK TYPE CLASSIFICATION ===
    # Comprehensive track classification based on characteristics

    track_types = {
        'street': [
            'Monaco', 'Monte Carlo',
            'Singapore', 'Marina Bay',
            'Baku', 'Azerbaijan',
            'Jeddah', 'Saudi', 'Corniche',
            'Miami',
            'Las Vegas',
        ],
        'power': [
            'Monza', 'Italian',
            'Spa', 'Belgian',
            'Silverstone', 'British',
            'Red Bull Ring', 'Austrian',
            'Interlagos', 'São Paulo', 'Brazilian',
        ],
        'technical': [
            'Hungary', 'Hungarian', 'Hungaroring',
            'Catalunya', 'Spanish', 'Barcelona',
            'Zandvoort', 'Dutch',
            'Suzuka', 'Japanese',
        ],
        'mixed': [
            'Bahrain', 'Sakhir',
            'Austin', 'United States', 'Circuit of the Americas', 'COTA',
            'Melbourne', 'Australian', 'Albert Park',
            'Shanghai', 'Chinese',
            'Imola', 'Emilia Romagna',
            'Montreal', 'Canadian', 'Gilles Villeneuve',
            'Mexico', 'Mexican', 'Hermanos Rodriguez',
            'Abu Dhabi', 'Yas Marina',
            'Qatar', 'Losail',
        ]
    }

    # Initialize all track type flags
    is_street_circuit = 0
    is_power_circuit = 0
    is_technical_circuit = 0
    is_mixed_circuit = 0

    # Get location and event name (handle None/NaN safely)
    location_str = str(location).lower() if location else ''
    event_name = ''
    if event is not None:
        if isinstance(event, dict):
            event_name = event.get('EventName', '')
        elif hasattr(event, 'get'):  # pandas Series
            try:
                event_name = event.get('EventName', '')
            except:
                event_name = ''
    event_name_str = str(event_name).lower()

    # Classify track (find first match)
    track_classified = False

    for circuit_type, keywords in track_types.items():
        if track_classified:
            break
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in location_str or keyword_lower in event_name_str:
                if circuit_type == 'street':
                    is_street_circuit = 1
                elif circuit_type == 'power':
                    is_power_circuit = 1
                elif circuit_type == 'technical':
                    is_technical_circuit = 1
                elif circuit_type == 'mixed':
                    is_mixed_circuit = 1
                track_classified = True
                break

    # If not classified, default to mixed (safest assumption)
    if not track_classified:
        is_mixed_circuit = 1
        print(f"  ⚠️  Track '{location}' not classified, defaulting to mixed")

    
    # === GRID FROM QUALI ===
    if not quali.empty:
        best_quali = quali.groupby("Driver")["LapTimeSec"].min()
        q_rank = best_quali.rank(method="min").astype(int)
        q_delta_to_p1 = best_quali - best_quali.min()
    else:
        q_rank = pd.Series(index=drivers, dtype=float)
        q_delta_to_p1 = pd.Series(index=drivers, dtype=float)
  
    # === Fetching the actual podium label (TRAINING PURPOSES ONLY) ===
    podium_label = pd.Series(index=drivers, dtype=float)  # ← ALWAYS define
    if is_training and race_completed and not race.empty:  # NEW: Safeguard with race_completed (prevents leakage)
        try:
            results = race.session.results
            if results is not None:
                pos = results.set_index("Abbreviation")["Position"]
                finish_position = pos.reindex(drivers, fill_value=21)  # DNF = 21
        except Exception:
            pass
    
    features = []
    for driver in drivers:
        d_fp2 = fp2[fp2["Driver"] == driver]
        d_fp3 = fp3[fp3["Driver"] == driver]
        d_quali = quali[quali["Driver"] == driver]

        # === QUALI-BASED PREDICTORS ===
        q_best = d_quali["LapTimeSec"].min() if not d_quali.empty else np.nan
        quali_sim = np.nan
        if not d_fp3.empty:
            soft = d_fp3[d_fp3["Compound"] == "SOFT"]
            # We are looking at a short stint on softs and taking a rolling minimum as the infered quali pace in hand
            soft = d_fp3[d_fp3["Compound"] == "SOFT"]
            if len(soft) >= 2:  # Relax to 2 laps
                quali_sim = soft["LapTimeSec"].min()  # Just take best lap
            elif not d_fp3.empty:
                # Fallback: use fastest lap on any compound
                quali_sim = d_fp3["LapTimeSec"].min()
            else:
                quali_sim = np.nan

        # === RACE-PACE PREDICTORS ===
        race_compounds = d_fp2[d_fp2["Compound"].isin(["MEDIUM", "HARD"])].copy()
        race_pace_best = consistency = deg_best = window_laps_best = best_compound = np.nan

        if not race_compounds.empty:
            best_pace = np.inf
            for compound in ["MEDIUM", "HARD"]:
                laps = race_compounds[race_compounds["Compound"] == compound]["LapTimeSec"]
                if len(laps) >= 5:
                    rolling = laps.rolling(5, min_periods=5).median()
                    min_pace = rolling.min()
                    if min_pace < best_pace:
                        best_pace = min_pace
                        best_compound = compound
                        race_pace_best = min_pace
                        window_laps_best = len(laps)

                        if len(laps) >= 5: #Relaxing deg threshold from 8 to 5 laps
                            X = np.arange(len(laps)).reshape(-1, 1)
                            y = laps.values
                            try:
                                deg_best = LinearRegression().fit(X, y).coef_[0]
                            except:
                                deg_best = np.nan
                        else:
                            # Fallback: estimate from first/last lap comparison
                            if len(laps) >= 3:
                                first_lap = laps.iloc[:2].mean()
                                last_lap = laps.iloc[-2:].mean()
                                deg_best = (last_lap - first_lap) / len(laps)  # Rough estimate
                            else:
                                deg_best = np.nan

                        iqr = laps.rolling(5, min_periods=5).apply(
                            lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True
                        )
                        consistency = iqr.min() if not iqr.empty else np.nan

        
                

        # Field wide pace markers:
        field_best = fp2[fp2["Compound"].isin(["MEDIUM", "HARD"])]["LapTimeSec"].median()
        pace_vs_field_best = race_pace_best - field_best if not pd.isna(race_pace_best) else np.nan

        # === TEAMMATE DELTA ===
        team = d_fp2["Team"].iloc[0] if not d_fp2.empty else "Unknown"
        teammate_fp2 = fp2[(fp2["Team"] == team) & (fp2["Driver"] != driver)]
        teammate_quali = quali[(quali["Team"] == team) & (quali["Driver"] != driver)]

        # Race Pace Delta:
        teammate_pace = np.nan
        if not teammate_fp2.empty:
            t_laps = teammate_fp2[teammate_fp2["Compound"].isin(["MEDIUM", "HARD"])]["LapTimeSec"]
            if len(t_laps) >= 5:
                teammate_pace = t_laps.rolling(5, min_periods=5).median().min()
        teammate_delta_race = race_pace_best - teammate_pace if not pd.isna(race_pace_best) and not pd.isna(teammate_pace) else np.nan

        # Quali delta:
        teammate_q_best = teammate_quali["LapTimeSec"].min() if not teammate_quali.empty else np.nan
        teammate_delta_quali = q_best - teammate_q_best if not pd.isna(q_best) and not pd.isna(teammate_q_best) else np.nan

        # === FORM FEATURES (Season Momentum) ===
        form_df = get_driver_form(season, round_no, driver)
        track_history_df = get_driver_track_history(driver, location, season)
        
        
        form_avg_quali_rank_l5 = form_df["quali_rank"].mean() if not form_df.empty else np.nan
        form_avg_finish_pos_l5 = form_df["finish_position"].mean() if not form_df.empty else np.nan
        form_avg_grid_delta_l5 = form_df["grid_to_finish_delta"].mean() if not form_df.empty else np.nan
        form_podium_rate_l5 = (form_df["finish_position"] <= 3).mean() if not form_df.empty else GLOBAL_PODIUM_RATE  # NEW: Consistent global imputation
        form_dnf_rate_l5 = (form_df["finish_position"] == 21).mean() if not form_df.empty else GLOBAL_DNF_RATE      # NEW: Consistent global imputation

        # NEW: Use races_in_window from updated get_driver_form
        races_in_window = form_df['races_in_window'].iloc[0] if 'races_in_window' in form_df.columns and not form_df.empty else 0

        track_avg_finish_pos = track_history_df["finish_position"].mean() if not track_history_df.empty else np.nan
        track_avg_grid_delta = track_history_df["grid_to_finish_delta"].mean() if not track_history_df.empty else np.nan
        track_podium_rate = (track_history_df["finish_position"] <= 3).mean() if not track_history_df.empty else GLOBAL_PODIUM_RATE  # NEW
        track_dnf_rate = (track_history_df["finish_position"] == 21).mean() if not track_history_df.empty else GLOBAL_DNF_RATE 

        # === TIRE STRATEGY FEATURES ===
        # Count laps per compound in FP2
        tire_usage = {}
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            compound_laps = d_fp2[d_fp2['Compound'] == compound]
            tire_usage[compound] = len(compound_laps)

        # Tire versatility (how many different compounds tried?)
        compounds_tried = d_fp2['Compound'].nunique()

        # Longest stint (can they do a long run?)
        if not d_fp2.empty:
            # Group consecutive laps on same compound
            compound_changes = (d_fp2['Compound'] != d_fp2['Compound'].shift()).cumsum()
            stint_lengths = d_fp2.groupby(compound_changes).size()
            longest_stint = stint_lengths.max()
        else:
            longest_stint = 0

        # Compound preference (which tire was fastest?)
        compound_pace = {}
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            comp_laps = d_fp2[d_fp2['Compound'] == compound]['LapTimeSec']
            if len(comp_laps) >= 3:
                compound_pace[compound] = comp_laps.median()

        # === Weather ===
        fp2_session = load_session(season, round_no, "FP2")
        weather = get_track_conditions(fp2_session)
        temp_fp2 = weather['air_temp']
        track_temp_fp2 = weather['track_temp']
        humidity_fp2 = weather['humidity']
        pressure_fp2 = weather['pressure']
        wind_speed_fp2 = weather['wind_speed']
        is_raining = weather['rainfall']

        # Temperature deviation from optimal (28-32°C is ideal)
        if pd.notna(temp_fp2):
            temp_deviation = abs(temp_fp2 - 30)  # Distance from 30°C optimal
        else:
            temp_deviation = np.nan

        # Weather variability (compare FP1 to FP2 for stability)
        fp1_session = load_session(season, round_no, "FP1")
        fp1_weather = get_track_conditions(fp1_session)
        weather_stability = 0.0
        if pd.notna(fp1_weather['air_temp']) and pd.notna(temp_fp2):
            weather_stability = abs(fp1_weather['air_temp'] - temp_fp2)  # >5°C change = unstable

        # Rain experience (did driver practice in wet?)
        wet_practice_laps = 0
        if is_raining or weather['rainfall']:
            wet_practice_laps = len(d_fp2)  # Count of laps in rain

        
        # === ADVANCED INTERACTIONS ===
        # These capture F1-specific synergies

        # 2. QUALI × FORM (good grid + good form = reliable performer)
        if pd.notna(q_rank.get(driver, np.nan)) and pd.notna(form_avg_finish_pos_l5):
            quali_form_strength = (1 / (q_rank.get(driver, np.nan) + 1)) * (1 / (form_avg_finish_pos_l5 + 1))
        else:
            quali_form_strength = np.nan

        # 3. TRACK HISTORY × CURRENT FORM (specialist in form = dangerous)
        if pd.notna(track_podium_rate) and pd.notna(form_podium_rate_l5):
            track_form_synergy = track_podium_rate * form_podium_rate_l5
        else:
            track_form_synergy = np.nan


        # 5. TIRE VERSATILITY × WEATHER (adaptable drivers in variable conditions)
        if pd.notna(compounds_tried) and pd.notna(weather_stability):
            adaptability_score = compounds_tried * (1 + weather_stability / 5)
        else:
            adaptability_score = np.nan

        # 6. TEAMMATE ADVANTAGE (beating teammate in both sessions)
        if pd.notna(teammate_delta_race) and pd.notna(teammate_delta_quali):
            teammate_dominance = int(teammate_delta_race < 0 and teammate_delta_quali < 0)
        else:
            teammate_dominance = 0

        # === TRACK-SPECIFIC SKILL AMPLIFIERS ===
        # Different tracks reward different attributes

        # Street circuits: Qualifying is king, consistency critical (walls = DNF risk)
        street_quali_importance = q_rank.get(driver, 20) * is_street_circuit
        street_consistency_reward = (1 / (consistency + 0.1)) * is_street_circuit if pd.notna(consistency) else 0

        # Power circuits: Raw pace matters, overtaking easier (grid position less critical)
        power_pace_importance = (1 / (race_pace_best + 0.1)) * is_power_circuit if pd.notna(race_pace_best) else 0
        power_overtaking_potential = form_avg_grid_delta_l5 * is_power_circuit if pd.notna(form_avg_grid_delta_l5) else 0

        # Technical circuits: Quali + race pace balance, tire management crucial
        technical_quali_pace_balance = (q_rank.get(driver, 20) + (race_pace_best if pd.notna(race_pace_best) else 90)) * is_technical_circuit
        technical_tire_management = (1 / (deg_best + 0.01)) * is_technical_circuit if pd.notna(deg_best) else 0

        # Mixed circuits: All skills matter equally
        mixed_balanced_skill = (q_rank.get(driver, 20) * 0.5 + form_avg_finish_pos_l5 * 0.5) * is_mixed_circuit if pd.notna(form_avg_finish_pos_l5) else q_rank.get(driver, 20) * is_mixed_circuit


        features.append({
            "Driver": driver,
            "season": season,
            "round": round_no,
            "location": location,
            "team": team,
            "is_sprint_weekend": int(is_sprint_weekend),

            # Qualifying inferences and results (Most signal)
            "quali_rank": q_rank.get(driver, np.nan),
            "quali_delta_to_p1": q_delta_to_p1.get(driver, np.nan),
            "quali_sim_fp3": quali_sim,
            "teammate_delta_quali": teammate_delta_quali,

            "race_pace_fp2_best": race_pace_best,
            "pace_vs_field_fp2": pace_vs_field_best,
            "consistency_iqr_fp2": consistency,
            "deg_best_per_lap": deg_best,
            "long_run_laps_best": window_laps_best, #*
            "teammate_delta_race": teammate_delta_race,

            # FORM
            "form_avg_quali_rank_l5": form_avg_quali_rank_l5,
            "form_avg_finish_pos_l5": form_avg_finish_pos_l5,
            "form_avg_grid_delta_l5": form_avg_grid_delta_l5,
            "form_podium_rate_l5": form_podium_rate_l5,
            "form_dnf_rate_l5": form_dnf_rate_l5, #*
            "races_in_window": races_in_window,  # NEW: Added for form representativeness

            # TRACK
            "track_avg_finish_pos": track_avg_finish_pos,
            "track_avg_grid_delta": track_avg_grid_delta,
            "track_podium_rate": track_podium_rate,
            "track_dnf_rate": track_dnf_rate, #*

            # Tire strategy
            'medium_laps_fp2': tire_usage.get('MEDIUM', 0),
            'compounds_tried': compounds_tried,
            'longest_stint_fp2': longest_stint,

            # Weather
            "temp_fp2": temp_fp2,
            'temp_deviation': temp_deviation,
            'weather_stability': weather_stability,
            'wet_practice_laps': wet_practice_laps,

            # Advanced interactions
            'quali_form_strength': quali_form_strength,
            'track_form_synergy': track_form_synergy,
            'adaptability_score': adaptability_score,
            'teammate_dominance': teammate_dominance,
            
            # Track type flags (one-hot encoded, mutually exclusive)
            'is_street_circuit': is_street_circuit,
            'is_power_circuit': is_power_circuit,
            'is_technical_circuit': is_technical_circuit,
            'is_mixed_circuit': is_mixed_circuit,
            
            # Track-specific skill amplifiers
            'street_quali_importance': street_quali_importance,
            'street_consistency_reward': street_consistency_reward,
            'power_pace_importance': power_pace_importance,
            'power_overtaking_potential': power_overtaking_potential,
            'technical_quali_pace_balance': technical_quali_pace_balance,
            'technical_tire_management': technical_tire_management,
            'mixed_balanced_skill': mixed_balanced_skill,

            # Target
            "finish_position": finish_position.get(driver, np.nan) if is_training else np.nan})
            
    
    
    # === SMART IMPUTATION ===
    df = pd.DataFrame(features)

    # Imputation strategies by feature type
    imputation_map = {
        # Pace features: use median of the field
        'race_pace_fp2_best': df['race_pace_fp2_best'].median(),
        'pace_vs_field_fp2': 0.0,  # 0 = average vs field
        'consistency_iqr_fp2': df['consistency_iqr_fp2'].median(),
        'quali_sim_fp3': df['quali_sim_fp3'].median(),
        
        # Degradation: neutral (0)
        'deg_best_per_lap': 0.0,
        
        # Teammate deltas: neutral (0 = equal to teammate)
        'teammate_delta_race': 0.0,
        'teammate_delta_quali': 0.0,
        
        # Long run laps: median
        #'long_run_laps_best': df['long_run_laps_best'].median(),
        
        # Track history: global averages
        'track_avg_finish_pos': 11.0,  # Mid-field
        'track_avg_grid_delta': 0.0,   # Neutral grid movement
        'track_podium_rate': 0.15,     # 3/20 baseline
        #'track_dnf_rate': 0.10,        # 2/20 baseline
        
        # Tire strategy features
        'medium_laps_fp2': df['medium_laps_fp2'].median(),
        'compounds_tried': 2.0,  # Typical: 2 compounds
        'longest_stint_fp2': df['longest_stint_fp2'].median(),
        
        # Weather features
        'temp_fp2': df['temp_fp2'].median(),
        #'track_temp_fp2': df['track_temp_fp2'].median(),
        #'humidity_fp2': df['humidity_fp2'].median(),
        #'pressure_fp2': df['pressure_fp2'].median(),
        #'wind_speed_fp2': df['wind_speed_fp2'].median(),
        'temp_deviation': df['temp_deviation'].median(),
        #'track_temp_stress': 0.0,  # Neutral stress
        'weather_stability': 2.0,  # Small change typical
        'wet_practice_laps': 0.0,  # Usually dry
        
        # Base interaction features (will be recalculated below)
        #'deg_x_temp': 0.0,
        #'pace_consistency': 0.0,
        'quali_form_strength': 0.0,
        'track_form_synergy': 0.0,
        #'deg_temp_impact': 0.0,
        'adaptability_score': 0.0,
        'teammate_dominance': 0,

        # Track-specific features (already 0 or calculated, just ensure filled)
        'street_quali_importance': 0.0,
        'street_consistency_reward': 0.0,
        'power_pace_importance': 0.0,
        'power_overtaking_potential': 0.0,
        'technical_quali_pace_balance': 0.0,
        'technical_tire_management': 0.0,
        'mixed_balanced_skill': 0.0,
    }

    # Apply imputation
    for col, fill_value in imputation_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    # === POST-IMPUTATION: Recalculate interaction features ===
    # Now all base features are filled, so interactions won't have NaN

    # 3. quali_form_strength (already calculated before imputation, just ensure filled)
    if 'quali_form_strength' in df.columns:
        df['quali_form_strength'] = df['quali_form_strength'].fillna(0.0)

    # 4. track_form_synergy (already calculated, ensure filled)
    if 'track_form_synergy' in df.columns:
        df['track_form_synergy'] = df['track_form_synergy'].fillna(0.0)

    # 6. adaptability_score (already calculated, ensure filled)
    if 'adaptability_score' in df.columns:
        df['adaptability_score'] = df['adaptability_score'].fillna(0.0)

    # 7. teammate_dominance (already calculated, ensure filled)
    if 'teammate_dominance' in df.columns:
        df['teammate_dominance'] = df['teammate_dominance'].fillna(0)

    return df