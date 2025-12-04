import pandas as pd
import numpy as np
from src.data_prep import  get_clean_laps, get_driver_form, get_driver_track_history, get_event_entrants, load_session, get_track_conditions
from typing import Optional, Tuple
from sklearn.linear_model import LinearRegression
# This is a backup of a working features draft 24/11/25
# NEW: Global imputation defaults (computed from historical data or approximated; update with real vals later)
GLOBAL_PODIUM_RATE = 0.15  # Approx 3/20 drivers podium per race
GLOBAL_DNF_RATE = 0.10     # Approx 2/20 DNF per race

def build_features(practice: pd.DataFrame, quali: pd.DataFrame, race: pd.DataFrame, event: pd.DataFrame, is_training: bool = False, race_completed: bool = False) -> pd.DataFrame:  # Added race_completed flag
    if practice.empty and quali.empty:
        return pd.DataFrame()
    
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
            if len(soft) >= 3:
                rolling_min = soft["LapTimeSec"].rolling(3, min_periods=3).min()
                quali_sim = rolling_min.min()

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

                        if len(laps) >= 8:
                            X = np.arange(len(laps)).reshape(-1, 1)
                            y = laps.values
                            deg_best = LinearRegression().fit(X, y).coef_[0]

                        iqr = laps.rolling(5, min_periods=5).apply(
                            lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True
                        )
                        consistency = iqr.min() if not iqr.empty else np.nan

        # === SOFT RACE PACE (FP2) ===
        soft_laps = d_fp2[d_fp2["Compound"] == "SOFT"]["LapTimeSec"]
        race_pace_soft = deg_soft = soft_run_laps = np.nan
        if len(soft_laps) >= 5:
            rolling = soft_laps.rolling(5, min_periods=5).median()
            race_pace_soft = rolling.min()
            soft_run_laps = len(soft_laps)

            if len(soft_laps) >= 8:
                X = np.arange(len(soft_laps)).reshape(-1, 1)
                y = soft_laps.values
                deg_soft = LinearRegression().fit(X, y).coef_[0]
                

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
        track_dnf_rate = (track_history_df["finish_position"] == 21).mean() if not track_history_df.empty else GLOBAL_DNF_RATE        # NEW

        # === Weather ===
        fp2_session = load_session(season, round_no, "FP2")
        weather = get_track_conditions(fp2_session)
        temp_fp2 = weather['air_temp']
        track_temp_fp2 = weather['track_temp']
        humidity_fp2 = weather['humidity']
        pressure_fp2 = weather['pressure']
        wind_speed_fp2 = weather['wind_speed']
        is_raining = weather['rainfall']


        features.append({
            "Driver": driver,
            "season": season,
            "round": round_no,
            "location": location,
            "team": team,

            # Qualifying inferences and results (Most signal)
            "quali_rank": q_rank.get(driver, np.nan),
            "quali_delta_to_p1": q_delta_to_p1.get(driver, np.nan),
            "quali_sim_fp3": quali_sim,
            "teammate_delta_quali": teammate_delta_quali,

            "race_pace_fp2_best": race_pace_best,
            "best_compound": best_compound,
            "pace_vs_field_fp2": pace_vs_field_best,
            "consistency_iqr_fp2": consistency,
            "deg_best_per_lap": deg_best,
            "long_run_laps_best": window_laps_best,
            "teammate_delta_race": teammate_delta_race,

            # SOFT FEATURES
            "race_pace_fp2_soft": race_pace_soft,
            "deg_soft_per_lap": deg_soft,
            "soft_run_laps": soft_run_laps,

            # FORM
            "form_avg_quali_rank_l5": form_avg_quali_rank_l5,
            "form_avg_finish_pos_l5": form_avg_finish_pos_l5,
            "form_avg_grid_delta_l5": form_avg_grid_delta_l5,
            "form_podium_rate_l5": form_podium_rate_l5,
            "form_dnf_rate_l5": form_dnf_rate_l5,
            "races_in_window": races_in_window,  # NEW: Added for form representativeness

            # TRACK
            "track_avg_finish_pos": track_avg_finish_pos,
            "track_avg_grid_delta": track_avg_grid_delta,
            "track_podium_rate": track_podium_rate,
            "track_dnf_rate": track_dnf_rate,

            # Weather
            "temp_fp2": temp_fp2,
            "track_temp_fp2": track_temp_fp2,
            "humidity_fp2": humidity_fp2,
            "pressure_fp2": pressure_fp2,
            "wind_speed_fp2": wind_speed_fp2,
            "is_raining": is_raining,
            "deg_x_temp": deg_soft * temp_fp2 if pd.notna(deg_soft) and pd.notna(temp_fp2) else np.nan,
            "hot_weather_flag": int(temp_fp2 > 32) if pd.notna(temp_fp2) else 0,

            # Target
            "finish_position": finish_position.get(driver, np.nan) if is_training else np.nan})
            
    
    
    return pd.DataFrame(features)