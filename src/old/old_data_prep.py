from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import fastf1
from fastf1.core import Session

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / ".fastf1cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

_event_cache: Dict[Tuple[int, int], dict] = {}
_session_cache = {}
_schedule_cache = {}


def get_event(season: int, roundno: int) -> dict:
    """Cached event metadata."""
    key = (season, roundno)
    if key not in _event_cache:
        _event_cache[key] = fastf1.get_event(season, roundno)
    return _event_cache[key]

def load_session(season: int, roundno: int, kind: str):
    """Load and cache a session."""
    s = fastf1.get_session(season, roundno, kind)
    s.load(telemetry=False, weather=True, messages=False)
    return s

def get_clean_laps(season: int, roundno: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, int]:  # Added group_size return
    sessions = {}
    for kind in ['FP1', 'FP2', 'FP3', 'Q', 'R']:
        try:
            s = load_session(season, roundno, kind)
            laps = s.laps.copy()
            laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
            laps["Session"] = kind

            # === SAFE Deleted handling ===
            deleted_mask = pd.Series([False] * len(laps), index=laps.index)
            if "Deleted" in laps.columns:
                deleted_mask = laps["Deleted"].fillna(False)

            clean = laps[
                laps["LapTimeSec"].notna() &
                laps["PitOutTime"].isna() &
                laps["PitInTime"].isna() &
                ~deleted_mask
            ].copy()

            clean["Compound"] = clean["Compound"].fillna("UNKNOWN")
            sessions[kind] = clean
            print(f"{kind}: {len(clean)} clean laps")

        except Exception as e:
            print(f"{kind} failed: {e}")
            sessions[kind] = pd.DataFrame()

    # Combine practice
    practice = pd.concat([
        sessions.get('FP1', pd.DataFrame()),
        sessions.get('FP2', pd.DataFrame()),
        sessions.get('FP3', pd.DataFrame())
    ], ignore_index=True)

    # NEW: Dynamic group size based on entrants (fixes hard-coded 20 assumption)
    entrants = get_event_entrants(season, roundno)
    group_size = len(entrants)

    return (
        practice,
        sessions.get('Q', pd.DataFrame()),
        sessions.get('R', pd.DataFrame()),
        get_event(season, roundno),
        group_size  # Return for use in training groups
    )

def get_all_clean_laps(season: int, roundno: int) -> pd.DataFrame:
    """Return all clean laps from FP2, FP3, Q, R."""
    laps_list = []
    for kind in ['FP2', 'FP3', 'Q', 'R']:
        try:
            s = load_session(season, roundno, kind)
            laps = s.laps.copy()
            laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
            laps["Session"] = kind
            clean = laps[
                laps["LapTimeSec"].notna() &
                laps["PitOutTime"].isna() &
                laps["PitInTime"].isna()
            ][["Driver", "LapTimeSec", "Compound", "Session", "LapNumber", "TrackStatus"]]
            laps_list.append(clean)
        except Exception as e:
            print(f"{kind} failed: {e}")
    return pd.concat(laps_list, ignore_index=True) if laps_list else pd.DataFrame()

def get_driver_form(season:int, current_round:int, driver:str, lookback:int = 5) -> pd.DataFrame:
    form_data = []
    rounds = list(range(max(1, current_round - lookback), current_round))

    # --- Terms used to classify status strings (case-insensitive contains) ---
    NON_CLASS_TERMS = [
        'DNF','DNS','DSQ','DQ','DISQUALIFIED','EXCLUDED','RETIRED','RET',
        'NOT CLASSIFIED','NC','WITHDRAWN','W/D','DID NOT START'
    ]
    CLASS_TERMS = ['FINISHED','LAP','LAPS','LAPPED','CLASSIFIED']

    for rnd in rounds:
        try:
            race = _get_race_session(season, rnd, 'R')
            if race is None:
                continue

            event = race.event  # Session already knows its event
            results = race.results
            if results is None or driver not in results['Abbreviation'].values:
                continue

            r_by_abbr = results.set_index('Abbreviation')
            # raw position (may be float)
            pos_raw = r_by_abbr.loc[driver, 'Position']
            pos = int(pos_raw) if pd.notna(pos_raw) else None

            # --- robust classification (DNF/DNS/etc -> 21; last classified -> 20) ---
            status_series = results['Status'].astype(str).str.upper()

            # build masks once for the whole race
            non_class_mask = None
            for t in NON_CLASS_TERMS:
                m = status_series.str.contains(t, case=False, regex=False)
                non_class_mask = m if non_class_mask is None else (non_class_mask | m)
            non_class_mask = non_class_mask.fillna(False) if non_class_mask is not None else status_series == ''

            white_class_mask = None
            for t in CLASS_TERMS:
                m = status_series.str.contains(t, case=False, regex=False)
                white_class_mask = m if white_class_mask is None else (white_class_mask | m)
            white_class_mask = white_class_mask.fillna(False) if white_class_mask is not None else status_series == ''

            classified_mask = white_class_mask & (~non_class_mask)
            max_classified_pos = int(results.loc[classified_mask, 'Position'].max()) if classified_mask.any() else None

            # driver's own classification
            status = str(r_by_abbr.loc[driver, 'Status']).upper()
            driver_non_class = any(term in status for term in NON_CLASS_TERMS)
            driver_white_class = any(term in status for term in CLASS_TERMS)
            driver_classified = (not driver_non_class) and driver_white_class

            if not driver_classified:
                penalized_pos = 21.0
            else:
                if max_classified_pos is not None and pos == max_classified_pos:
                    penalized_pos = 20.0
                else:
                    penalized_pos = float(pos)

            # --- Quali results / grid position (safe if missing) ---
            quali = fastf1.get_session(season, rnd, 'Q')
            quali.load(telemetry=False)
            q_results = quali.results

            if q_results is not None and driver in q_results['Abbreviation'].values:
                q_by_abbr = q_results.set_index('Abbreviation')
                quali_rank = float(q_by_abbr.loc[driver, 'Position'])
                grid_pos = int(q_by_abbr.loc[driver, 'Position'])
            else:
                quali_rank = np.nan
                grid_pos = np.nan

            # delta only if both are numbers
            if pd.notna(grid_pos) and penalized_pos is not None:
                grid_to_finish_delta = float(grid_pos - penalized_pos)
            else:
                grid_to_finish_delta = np.nan

            form_data.append({
                'Driver': driver,
                'season': season,
                'round': rnd,
                'quali_rank': quali_rank,
                'grid_to_finish_delta': grid_to_finish_delta,
                'finish_position': penalized_pos
            })

        except Exception:
            continue

    form_df = pd.DataFrame(form_data)
    # NEW: Add races_in_window for better representativeness (as per feedback)
    if not form_df.empty:
        form_df['races_in_window'] = len(form_df)
        # Optional recency weight (applied later in features if needed)
        form_df['recency_weight'] = np.linspace(0.5, 1.0, len(form_df))  # More weight to recent races

    return form_df

def get_driver_track_history(driver: str, location: str, year: int, max_years: int = 5) -> pd.DataFrame:
    """
    Fetch driver results for a given location across recent seasons.
    Penalize DNF/DNS/NC/retirements (anything non-classified) to 21.
    If driver is last classified finisher, penalize to 20.
    Includes rich debug prints so we can see how statuses are interpreted per year.
    """
    current_year = year
    years = list(range(max(2022,current_year - max_years + 1), current_year + 1))  # NEW: Include current year if applicable (fixes exclusion)
    track_data = []
    

    for year in years:
        try:
            schedule = _get_schedule(year)
            if schedule is None:
                continue

            event_row = schedule[schedule['Location'].str.contains(location, case=False, na=False)]
            if event_row.empty:
                continue

            rnd = int(event_row.iloc[0]['RoundNumber'])

            race = _get_race_session(year, rnd, 'R')
            if race is None:
                continue

            results = race.results
            if results is None or driver not in results['Abbreviation'].values:
                continue

            # --- DEBUG dump of unique statuses for this race once per loop ---
            uniq_statuses = (
                results['Status']
                .astype(str)
                .str.strip()
                .str.upper()
                .dropna()
                .unique()
                .tolist()
            )

            r_by_abbr = results.set_index('Abbreviation')
            raw_pos = r_by_abbr.loc[driver, 'Position']
            pos = int(raw_pos) if pd.notna(raw_pos) else None
            status_raw = str(r_by_abbr.loc[driver, 'Status'])
            status = status_raw.upper().strip()

            

            # Classify with two-stage logic: blacklist (non-classified) overrides whitelist (classified)
            status_series = results['Status'].astype(str).str.upper()

            non_terms = [
                'DNF','DNS','DSQ','DQ','DISQUALIFIED','EXCLUDED','RETIRED','RET',
                'NOT CLASSIFIED','NC','WITHDRAWN','W/D','DID NOT START'
            ]
            non_class_mask = None
            for t in non_terms:
                mask_t = status_series.str.contains(t, case=False, regex=False)
                non_class_mask = mask_t if non_class_mask is None else (non_class_mask | mask_t)

            white_terms = ['FINISHED','LAP','LAPS','LAPPED','CLASSIFIED']
            white_class_mask = None
            for t in white_terms:
                mask_t = status_series.str.contains(t, case=False, regex=False)
                white_class_mask = mask_t if white_class_mask is None else (white_class_mask | mask_t)

            non_class_mask = non_class_mask.fillna(False) if non_class_mask is not None else status_series == ''
            white_class_mask = white_class_mask.fillna(False) if white_class_mask is not None else status_series == ''

            classified_mask = white_class_mask & (~non_class_mask)
            max_classified_pos = int(results.loc[classified_mask, 'Position'].max()) if classified_mask.any() else None

            # Driver's own classification
            driver_non_class = any(term in status for term in non_terms)
            driver_white_class = any(term in status for term in white_terms)
            driver_classified = (not driver_non_class) and driver_white_class

            if not driver_classified:
                penalized_pos = 21.0
            else:
                if max_classified_pos is not None and pos == max_classified_pos:
                    penalized_pos = 20.0
                else:
                    penalized_pos = float(pos)

            # Quali (grid) position
            try:
                quali = fastf1.get_session(year, rnd, 'Q')
                quali.load(telemetry=False)
                q_results = quali.results
                grid_pos = int(q_results.set_index('Abbreviation').loc[driver, 'Position'])
            except Exception as e_q:
                grid_pos = None

            grid_to_finish_delta = float(grid_pos - penalized_pos) if (grid_pos is not None and penalized_pos is not None) else None

            track_data.append({
                'Driver': driver,
                'season': year,
                'location': location,
                'grid_to_finish_delta': grid_to_finish_delta,
                'finish_position': penalized_pos
            })

        except Exception as e:
            continue

    return pd.DataFrame(track_data)


def get_event_entrants(season: int, round_no: int) -> set[str]:
    """
    Prefer race entrants; if race not loaded/available (pre-race), fall back to Q.
    """
    entrants = set()
    try:
        sess = fastf1.get_session(season, round_no, 'R')
        sess.load(telemetry=False, weather=False, messages=False)
        if sess.results is not None and not sess.results.empty:
            entrants = set(sess.results['Abbreviation'].dropna().unique())
    except Exception:
        pass

    if not entrants:
        try:
            q = fastf1.get_session(season, round_no, 'Q')
            q.load(telemetry=False, weather=False, messages=False)
            if q.results is not None and not q.results.empty:
                entrants = set(q.results['Abbreviation'].dropna().unique())
        except Exception:
            pass

    return entrants


def get_track_conditions(
    session: Session,
    include_weather: bool = True,
    include_track_temp: bool = True
) -> dict:
    """
    Extract track conditions from a FastF1 session.
    
    Returns:
        dict with:
            - air_temp: mean AirTemp
            - track_temp: mean TrackTemp
            - humidity: mean Humidity
            - pressure: mean Pressure
            - wind_speed: mean WindSpeed
            - rainfall: bool (any rain?)
    """
    conditions = {
        'air_temp': np.nan,
        'track_temp': np.nan,
        'humidity': np.nan,
        'pressure': np.nan,
        'wind_speed': np.nan,
        'rainfall': False
    }

    if session is None:
        return conditions

    try:
        weather = session.weather_data
        if weather is not None and not weather.empty:
            if include_weather and 'AirTemp' in weather.columns:
                conditions['air_temp'] = weather['AirTemp'].mean()
            if include_weather and 'Humidity' in weather.columns:
                conditions['humidity'] = weather['Humidity'].mean()
            if include_weather and 'Pressure' in weather.columns:
                conditions['pressure'] = weather['Pressure'].mean()
            if include_weather and 'WindSpeed' in weather.columns:
                conditions['wind_speed'] = weather['WindSpeed'].mean()
            if include_weather and 'Rainfall' in weather.columns:
                conditions['rainfall'] = bool(weather['Rainfall'].any())

            if include_track_temp and 'TrackTemp' in weather.columns:
                conditions['track_temp'] = weather['TrackTemp'].mean()
    except Exception as e:
        print(f"Weather data failed: {e}")

    # NEW: Fallback to race weather if FP2 data missing (as per feedback)
    if pd.isna(conditions['air_temp']) and session.session.kind != 'R':  # Avoid infinite loop
        try:
            race_session = load_session(session.event['Season'], session.event['RoundNumber'], 'R')
            fallback_conditions = get_track_conditions(race_session, include_weather, include_track_temp)
            for key in conditions:
                if pd.isna(conditions[key]):
                    conditions[key] = fallback_conditions[key]
        except Exception:
            pass

    return conditions

def _get_race_session(season: int, rnd: int, kind: str = 'R') -> fastf1.core.Session | None:
    """
    Return a loaded FastF1 session from an in-process cache.
    This sits on top of FastF1's own cache to avoid repeated load() calls.
    """
    key = (season, rnd, kind)
    if key in _session_cache:
        return _session_cache[key]

    try:
        sess = fastf1.get_session(season, rnd, kind)
        sess.load(telemetry=False, weather=False, messages=False)
    except Exception:
        return None

    _session_cache[key] = sess
    return sess

def _get_schedule(year: int) -> pd.DataFrame | None:
    """
    Return event schedule for a year with an in-process cache.
    """
    if year in _schedule_cache:
        return _schedule_cache[year]
    try:
        sched = fastf1.get_event_schedule(year)
    except Exception:
        return None
    _schedule_cache[year] = sched
    return sched