import argparse
import sys
from pathlib import Path
import fastf1
#NOTE TO SELF: use grok_features, grok_model, and normal data_prep, warm_cache, train_all_races
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep import (
    get_clean_laps, 
    get_event_entrants, 
    get_driver_form, 
    get_driver_track_history,
    load_session,
    _get_schedule,
    CACHE_DIR
)

# Enable cache immediately
fastf1.Cache.enable_cache(str(CACHE_DIR))


def validate_cache_health():
    """Check if cache directory is healthy."""
    if not CACHE_DIR.exists():
        print(f"❌ Cache directory doesn't exist: {CACHE_DIR}")
        return False
    
    cache_files = list(CACHE_DIR.glob("*"))
    print(f"✅ Cache directory exists with {len(cache_files)} files")
    return True


def warm_schedule(season: int) -> bool:
    """Warm the season schedule."""
    print(f"  📅 Warming schedule for {season}...")
    try:
        schedule = _get_schedule(season)
        if schedule is None or schedule.empty:
            print(f"    ❌ Schedule empty for {season}")
            return False
        print(f"    ✅ Schedule loaded: {len(schedule)} rounds")
        return True
    except Exception as e:
        print(f"    ❌ Schedule failed: {e}")
        return False


def warm_race(season: int, rnd: int, warm_history: bool = True) -> dict:
    """
    Warm cache for a single race weekend.
    Returns stats about what was loaded.
    """
    stats = {
        'season': season,
        'round': rnd,
        'sessions_ok': [],
        'sessions_failed': [],
        'drivers_warmed': 0,
        'history_ok': True
    }
    
    print(f"\n--- Warming {season} R{rnd} ---")
    
    # 1. Load all sessions
    for kind in ['FP1', 'FP2', 'FP3', 'Q', 'R']:
        try:
            sess = load_session(season, rnd, kind)
            if sess is not None:
                print(f"  ✅ {kind} loaded")
                stats['sessions_ok'].append(kind)
            else:
                print(f"  ⚠️  {kind} returned None")
                stats['sessions_failed'].append(kind)
        except Exception as e:
            print(f"  ❌ {kind} failed: {e}")
            stats['sessions_failed'].append(kind)
    
    # 2. Load clean laps (this validates data structure)
    try:
        practice, quali, race, event, group_size = get_clean_laps(season, rnd)
        print(f"  ✅ get_clean_laps: practice={len(practice)}, quali={len(quali)}, race={len(race)}, group_size={group_size}")
    except Exception as e:
        print(f"  ❌ get_clean_laps failed: {e}")
        return stats
    
    if not warm_history:
        return stats
    
    # 3. Warm driver history
    # event is a pandas Series, so we need to check it properly
    if event is None or (hasattr(event, 'empty') and event.empty):
        location = "Unknown"
    else:
        location = event.get("Location", "Unknown")
    
    entrants = get_event_entrants(season, rnd)
    
    if not entrants:
        print(f"  ⚠️  No entrants found")
        return stats
    
    print(f"  🏎️  Warming history for {len(entrants)} drivers...")
    drivers_ok = 0
    drivers_failed = 0
    
    for drv in sorted(entrants):
        form_ok = history_ok = False
        
        # Form
        try:
            form_df = get_driver_form(season, rnd, drv)
            form_ok = True
        except Exception as e:
            print(f"    ⚠️  Form failed for {drv}: {e}")
        
        # Track history
        try:
            hist_df = get_driver_track_history(drv, location, season)
            history_ok = True
        except Exception as e:
            print(f"    ⚠️  History failed for {drv}: {e}")
        
        if form_ok and history_ok:
            drivers_ok += 1
        else:
            drivers_failed += 1
    
    print(f"  ✅ Driver history: {drivers_ok} OK, {drivers_failed} failed")
    stats['drivers_warmed'] = drivers_ok
    stats['history_ok'] = (drivers_failed == 0)
    
    return stats


def discover_rounds(season: int) -> list[int]:
    """
    Discover available rounds by probing.
    More robust than relying on potentially corrupted schedule.
    """
    print(f"\n🔍 Discovering rounds for {season}...")
    available = []
    
    for rnd in range(1, 31):  # Probe up to round 30
        try:
            event = fastf1.get_event(season, rnd)
            if event is not None:
                available.append(rnd)
                print(f"  ✅ R{rnd}: {event.get('EventName', 'Unknown')}")
        except Exception:
            continue
    
    print(f"✅ Found {len(available)} rounds for {season}")
    return available


def main():
    parser = argparse.ArgumentParser(
        description="Warm FastF1 cache for offline training"
    )
    parser.add_argument(
        "--seasons", 
        nargs="+", 
        type=int, 
        required=True,
        help="Seasons to cache (e.g., 2022 2023 2024)"
    )
    parser.add_argument(
        "--rounds", 
        nargs="*", 
        type=int, 
        default=None,
        help="Specific rounds (if omitted, auto-discovers all)"
    )
    parser.add_argument(
        "--no-history", 
        action="store_true",
        help="Skip driver form/history (faster, sessions only)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Just check cache health, don't warm"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="⚠️  CLEAR cache before warming (nuclear option)"
    )
    
    args = parser.parse_args()
    
    # Validate cache
    if args.validate_only:
        validate_cache_health()
        return
    
    # Clear cache if requested
    if args.clear_cache:
        from src.data_prep import clear_corrupted_cache
        confirm = input("⚠️  This will DELETE all cached data. Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return
        clear_corrupted_cache()
    
    # Ensure cache dir exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    
    for season in args.seasons:
        print(f"\n{'='*60}")
        print(f"🏁 SEASON {season}")
        print(f"{'='*60}")
        
        # Warm schedule first
        if not warm_schedule(season):
            print(f"⚠️  Skipping {season} due to schedule failure")
            continue
        
        # Determine rounds
        if args.rounds:
            rounds = sorted(set(args.rounds))
            print(f"📋 Using specified rounds: {rounds}")
        else:
            rounds = discover_rounds(season)
            if not rounds:
                print(f"❌ No rounds discovered for {season}")
                continue
        
        # Warm each round
        for rnd in rounds:
            try:
                stats = warm_race(
                    season, 
                    rnd, 
                    warm_history=not args.no_history
                )
                all_stats.append(stats)
            except Exception as e:
                print(f"❌ Fatal error warming {season} R{rnd}: {e}")
                continue
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 WARMING SUMMARY")
    print(f"{'='*60}")
    
    total_races = len(all_stats)
    fully_ok = sum(1 for s in all_stats if len(s['sessions_failed']) == 0 and s['history_ok'])
    partial_ok = sum(1 for s in all_stats if len(s['sessions_ok']) > 0 and not (len(s['sessions_failed']) == 0 and s['history_ok']))
    failed = sum(1 for s in all_stats if len(s['sessions_ok']) == 0)
    
    print(f"Total races attempted: {total_races}")
    print(f"✅ Fully successful: {fully_ok}")
    print(f"⚠️  Partially successful: {partial_ok}")
    print(f"❌ Failed: {failed}")
    
    if fully_ok == total_races:
        print("\n🎉 ALL RACES CACHED SUCCESSFULLY!")
        print("✅ You can now train with --offline mode")
    elif fully_ok > 0:
        print(f"\n✅ {fully_ok}/{total_races} races ready for training")
        print("⚠️  Some races had issues - check logs above")
    else:
        print("\n❌ NO RACES SUCCESSFULLY CACHED")
        print("💡 Try: python warm_cache.py --clear-cache --seasons 2024 --rounds 1 2 3")
    
    print(f"\n💾 Cache location: {CACHE_DIR}")


if __name__ == "__main__":
    main()