from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import fastf1
from fastf1.core import Session
import warnings
warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / ".fastf1cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# CRITICAL: In-memory caches to prevent redundant API calls
_event_cache: Dict[Tuple[int, int], dict] = {}
_session_cache: Dict[Tuple[int, int, str], Optional[Session]] = {}
_schedule_cache: Dict[int, pd.DataFrame] = {}
_results_cache: Dict[Tuple[int, int, str], Optional[pd.DataFrame]] = {}


def clear_corrupted_cache():
    """
    Nuclear option: clear all FastF1 cache files.
    Use this if you suspect cache corruption.
    """
    import shutil
    if CACHE_DIR.exists():
        print(f"🗑️  Clearing cache at {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ Cache cleared. Re-run warm_cache.py to rebuild.")


def get_event(season: int, roundno: int) -> dict:
    """Cached event metadata with validation. Returns dict or None."""
    key = (season, roundno)
    if key not in _event_cache:
        try:
            event = fastf1.get_event(season, roundno)
            # Convert pandas Series to dict for safer handling
            if hasattr(event, 'to_dict'):
                _event_cache[key] = event.to_dict()
            else:
                _event_cache[key] = event
        except Exception as e:
            print(f"⚠️  get_event failed for {season} R{roundno}: {e}")
            _event_cache[key] = None
    return _event_cache[key]


def load_session(season: int, roundno: int, kind: str) -> Optional[Session]:
    """Load and cache a session with robust error handling."""
    key = (season, roundno, kind)
    
    if key in _session_cache:
        return _session_cache[key]
    
    try:
        s = fastf1.get_session(season, roundno, kind)
        s.load(telemetry=False, weather=True, messages=False)
        _session_cache[key] = s
        return s
    except Exception as e:
        print(f"⚠️  Session load failed {season} R{roundno} {kind}: {e}")
        _session_cache[key] = None
        return None


def get_clean_laps(season: int, roundno: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, int]:
    """
    GUARANTEED to return 5 values: (practice, quali, race, event, group_size)
    All DataFrames are guaranteed to exist (may be empty).
    """
    sessions = {}
    
    for kind in ['FP1', 'FP2', 'FP3', 'Q', 'R']:
        try:
            s = load_session(season, roundno, kind)
            if s is None:
                sessions[kind] = pd.DataFrame()
                continue
                
            laps = s.laps.copy()
            laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
            laps["Session"] = kind

            # Safe Deleted handling
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
            print(f"  {kind}: {len(clean)} clean laps")

        except Exception as e:
            print(f"  {kind} failed: {e}")
            sessions[kind] = pd.DataFrame()

    # Combine practice
    practice = pd.concat([
        sessions.get('FP1', pd.DataFrame()),
        sessions.get('FP2', pd.DataFrame()),
        sessions.get('FP3', pd.DataFrame())
    ], ignore_index=True)

    # Dynamic group size based on entrants
    entrants = get_event_entrants(season, roundno)
    group_size = len(entrants) if entrants else 20  # fallback to 20

    event = get_event(season, roundno)
    if event is None:
        event = {
            'EventDate': pd.NaT, 
            'RoundNumber': roundno, 
            'Location': 'Unknown',
            'Season': season
        }

    return (
        practice,
        sessions.get('Q', pd.DataFrame()),
        sessions.get('R', pd.DataFrame()),
        event,
        group_size
    )


def get_driver_form(season: int, current_round: int, driver: str, lookback: int = 5) -> pd.DataFrame:
    """
    Driver form from CACHED sessions only. No API calls.
    """
    form_data = []
    rounds = list(range(max(1, current_round - lookback), current_round))

    NON_CLASS_TERMS = [
        'DNF', 'DNS', 'DSQ', 'DQ', 'DISQUALIFIED', 'EXCLUDED', 'RETIRED', 'RET',
        'NOT CLASSIFIED', 'NC', 'WITHDRAWN', 'W/D', 'DID NOT START'
    ]
    CLASS_TERMS = ['FINISHED', 'LAP', 'LAPS', 'LAPPED', 'CLASSIFIED']

    for rnd in rounds:
        try:
            # Use cached session - no new API calls
            race_session = load_session(season, rnd, 'R')
            if race_session is None:
                continue

            results = _get_results(season, rnd, 'R')
            if results is None or driver not in results['Abbreviation'].values:
                continue

            r_by_abbr = results.set_index('Abbreviation')
            pos_raw = r_by_abbr.loc[driver, 'Position']
            pos = int(pos_raw) if pd.notna(pos_raw) else None

            # Classification logic
            status_series = results['Status'].astype(str).str.upper()
            
            non_class_mask = pd.Series([False] * len(results), index=results.index)
            for term in NON_CLASS_TERMS:
                non_class_mask |= status_series.str.contains(term, case=False, regex=False, na=False)

            white_class_mask = pd.Series([False] * len(results), index=results.index)
            for term in CLASS_TERMS:
                white_class_mask |= status_series.str.contains(term, case=False, regex=False, na=False)

            classified_mask = white_class_mask & (~non_class_mask)
            max_classified_pos = int(results.loc[classified_mask, 'Position'].max()) if classified_mask.any() else None

            status = str(r_by_abbr.loc[driver, 'Status']).upper()
            driver_non_class = any(term in status for term in NON_CLASS_TERMS)
            driver_white_class = any(term in status for term in CLASS_TERMS)
            driver_classified = (not driver_non_class) and driver_white_class

            if not driver_classified:
                penalized_pos = 21.0
            elif max_classified_pos is not None and pos == max_classified_pos:
                penalized_pos = 20.0
            else:
                penalized_pos = float(pos)

            # Quali data
            quali_results = _get_results(season, rnd, 'Q')
            if quali_results is not None and driver in quali_results['Abbreviation'].values:
                q_by_abbr = quali_results.set_index('Abbreviation')
                quali_rank = float(q_by_abbr.loc[driver, 'Position'])
                grid_pos = int(q_by_abbr.loc[driver, 'Position'])
            else:
                quali_rank = np.nan
                grid_pos = np.nan

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

        except Exception as e:
            continue

    form_df = pd.DataFrame(form_data)
    if not form_df.empty:
        form_df['races_in_window'] = len(form_df)
        form_df['recency_weight'] = np.linspace(0.5, 1.0, len(form_df))

    return form_df


def get_driver_track_history(driver: str, location: str, year: int, max_years: int = 5) -> pd.DataFrame:
    """
    Driver track history from CACHED data only (2022+). No API calls.
    """
    current_year = year
    years = list(range(max(2022, current_year - max_years + 1), current_year))
    track_data = []

    NON_TERMS = [
        'DNF', 'DNS', 'DSQ', 'DQ', 'DISQUALIFIED', 'EXCLUDED', 'RETIRED', 'RET',
        'NOT CLASSIFIED', 'NC', 'WITHDRAWN', 'W/D', 'DID NOT START'
    ]
    WHITE_TERMS = ['FINISHED', 'LAP', 'LAPS', 'LAPPED', 'CLASSIFIED']

    for yr in years:
        try:
            schedule = _get_schedule(yr)
            if schedule is None:
                continue

            event_row = schedule[schedule['Location'].str.contains(location, case=False, na=False)]
            if event_row.empty:
                continue

            rnd = int(event_row.iloc[0]['RoundNumber'])
            results = _get_results(yr, rnd, 'R')
            
            if results is None or driver not in results['Abbreviation'].values:
                continue

            r_by_abbr = results.set_index('Abbreviation')
            raw_pos = r_by_abbr.loc[driver, 'Position']
            pos = int(raw_pos) if pd.notna(raw_pos) else None
            status = str(r_by_abbr.loc[driver, 'Status']).upper().strip()

            # Classification
            status_series = results['Status'].astype(str).str.upper()
            
            non_class_mask = pd.Series([False] * len(results), index=results.index)
            for term in NON_TERMS:
                non_class_mask |= status_series.str.contains(term, case=False, regex=False, na=False)

            white_class_mask = pd.Series([False] * len(results), index=results.index)
            for term in WHITE_TERMS:
                white_class_mask |= status_series.str.contains(term, case=False, regex=False, na=False)

            classified_mask = white_class_mask & (~non_class_mask)
            max_classified_pos = int(results.loc[classified_mask, 'Position'].max()) if classified_mask.any() else None

            driver_non_class = any(term in status for term in NON_TERMS)
            driver_white_class = any(term in status for term in WHITE_TERMS)
            driver_classified = (not driver_non_class) and driver_white_class

            if not driver_classified:
                penalized_pos = 21.0
            elif max_classified_pos is not None and pos == max_classified_pos:
                penalized_pos = 20.0
            else:
                penalized_pos = float(pos)

            # Grid position
            quali_results = _get_results(yr, rnd, 'Q')
            grid_pos = None
            if quali_results is not None and driver in quali_results['Abbreviation'].values:
                grid_pos = int(quali_results.set_index('Abbreviation').loc[driver, 'Position'])

            grid_to_finish_delta = float(grid_pos - penalized_pos) if (grid_pos is not None) else None

            track_data.append({
                'Driver': driver,
                'season': yr,
                'location': location,
                'grid_to_finish_delta': grid_to_finish_delta,
                'finish_position': penalized_pos
            })

        except Exception:
            continue

    return pd.DataFrame(track_data)


def get_event_entrants(season: int, round_no: int) -> set[str]:
    """Get entrants from cached race or quali results."""
    entrants = set()
    
    # Try race first
    results = _get_results(season, round_no, 'R')
    if results is not None and not results.empty:
        entrants = set(results['Abbreviation'].dropna().unique())
    
    # Fallback to quali
    if not entrants:
        results = _get_results(season, round_no, 'Q')
        if results is not None and not results.empty:
            entrants = set(results['Abbreviation'].dropna().unique())
    
    return entrants


def get_track_conditions(session: Session, include_weather: bool = True, include_track_temp: bool = True) -> dict:
    """Extract track conditions with fallback to race session."""
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
        print(f"  Weather data failed: {e}")

    # Fallback to race weather if FP2 missing
    if pd.isna(conditions['air_temp']) and hasattr(session, 'event'):
        try:
            race_session = load_session(
                session.event['Season'], 
                session.event['RoundNumber'], 
                'R'
            )
            if race_session is not None:
                fallback = get_track_conditions(race_session, include_weather, include_track_temp)
                for key in conditions:
                    if pd.isna(conditions[key]) or conditions[key] is False:
                        conditions[key] = fallback[key]
        except Exception:
            pass

    return conditions


def _get_results(season: int, rnd: int, kind: str) -> Optional[pd.DataFrame]:
    """Get results from cached session with memoization."""
    key = (season, rnd, kind)
    
    if key in _results_cache:
        return _results_cache[key]
    
    session = load_session(season, rnd, kind)
    if session is None:
        _results_cache[key] = None
        return None
    
    try:
        results = session.results
        _results_cache[key] = results
        return results
    except Exception:
        _results_cache[key] = None
        return None


def _get_schedule(year: int) -> Optional[pd.DataFrame]:
    """Get schedule with caching."""
    if year in _schedule_cache:
        return _schedule_cache[year]
    
    try:
        sched = fastf1.get_event_schedule(year)
        _schedule_cache[year] = sched
        return sched
    except Exception as e:
        print(f"⚠️  Schedule load failed for {year}: {e}")
        return None