import os
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import fastf1

# Setup paths
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / ".fastf1cache"
sys.path.insert(0, str(Path(__file__).parent.parent))

# CRITICAL: Enable cache BEFORE imports
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Now import project modules
from src.data_prep import get_clean_laps, get_event
from src.features import build_features
from src.model import F1PodiumModel


def enable_offline_mode(strict: bool = True):
    """
    Force FastF1 into offline mode.
    If strict=True, will fail on any API attempt.
    """
    try:
        fastf1.Cache.offline_mode(enabled=True)
        print("✅ FastF1 offline mode enabled")
    except Exception as e:
        print(f"⚠️  Could not enable offline mode: {e}")
        if strict:
            raise


def discover_rounds_offline(season: int) -> list[int]:
    """
    Discover rounds by probing cached events (no API calls).
    """
    available = []
    for rnd in range(1, 31):
        try:
            event = fastf1.get_event(season, rnd)
            if event is not None:
                available.append(rnd)
        except Exception:
            continue
    return available


def validate_features(X: pd.DataFrame, y: pd.Series, groups: list[int]) -> bool:
    """
    Validate feature matrix before training.
    Returns True if valid, False otherwise.
    """
    issues = []
    
    # Check shapes
    if len(X) != len(y):
        issues.append(f"Shape mismatch: X={len(X)}, y={len(y)}")
    
    if sum(groups) != len(X):
        issues.append(f"Group sum mismatch: sum(groups)={sum(groups)}, len(X)={len(X)}")
    
    # Check for NaN in labels
    nan_labels = y.isna().sum()
    if nan_labels > 0:
        issues.append(f"NaN labels: {nan_labels} ({100*nan_labels/len(y):.1f}%)")
    
    # Check for infinite values
    if np.isinf(y).any():
        issues.append(f"Infinite labels: {np.isinf(y).sum()}")
    
    # Check label range
    valid_labels = y[y.notna()]
    if len(valid_labels) > 0:
        if valid_labels.min() < 1 or valid_labels.max() > 21:
            issues.append(f"Invalid label range: [{valid_labels.min()}, {valid_labels.max()}] (should be [1, 21])")
    
    # Feature checks
    nan_pct = X.isna().sum().sum() / (X.shape[0] * X.shape[1]) * 100
    print(f"📊 Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"📊 NaN in features: {nan_pct:.1f}%")
    print(f"📊 Groups: {len(groups)} races, avg {np.mean(groups):.1f} drivers/race")
    
    if issues:
        print("\n❌ VALIDATION FAILED:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    
    print("✅ Feature validation passed")
    return True


def collect_training_data(seasons: list[int], rounds: list[int] = None, offline: bool = True):
    """
    Collect training data with robust error handling.
    """
    all_feats = []
    groups = []
    race_keys = []
    skipped = []
    
    if offline:
        enable_offline_mode(strict=True)
    
    for season in seasons:
        # Discover or use provided rounds
        if rounds:
            rnds = sorted(set(rounds))
        else:
            rnds = discover_rounds_offline(season)
        
        if not rnds:
            print(f"⚠️  No rounds found for {season}")
            continue
        
        print(f"\n🏁 Processing {season} ({len(rnds)} rounds)")
        
        for rnd in tqdm(rnds, desc=f"Season {season}"):
            try:
                # Load data
                practice, quali, race, event, group_size = get_clean_laps(season, rnd)
                
                # Validate event
                if event is None:
                    skipped.append((season, rnd, "No event metadata"))
                    continue
                
                # Check if race completed
                race_completed = (race is not None and not race.empty)
                
                # Build features
                feats = build_features(
                    practice, 
                    quali, 
                    race, 
                    event,
                    is_training=True,
                    race_completed=race_completed
                )
                
                if feats is None or feats.empty:
                    skipped.append((season, rnd, "Empty features"))
                    continue
                
                # Check for labels
                if "finish_position" not in feats.columns:
                    skipped.append((season, rnd, "No finish_position column"))
                    continue
                
                # Remove rows with NaN labels
                valid_rows = feats["finish_position"].notna()
                n_invalid = (~valid_rows).sum()
                
                if n_invalid == len(feats):
                    skipped.append((season, rnd, "All labels NaN"))
                    continue
                
                if n_invalid > 0:
                    print(f"  ⚠️  R{rnd}: Dropping {n_invalid} rows with NaN labels")
                    feats = feats[valid_rows].reset_index(drop=True)
                
                # Validate group size
                actual_group_size = len(feats)
                if actual_group_size < 10:
                    skipped.append((season, rnd, f"Too few drivers ({actual_group_size})"))
                    continue
                
                all_feats.append(feats)
                groups.append(actual_group_size)
                race_keys.append((season, rnd))
                
            except Exception as e:
                skipped.append((season, rnd, str(e)))
                print(f"  ❌ R{rnd} failed: {e}")
                continue
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DATA COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully loaded: {len(groups)} races")
    print(f"❌ Skipped: {len(skipped)} races")
    
    if skipped:
        print("\nSkipped races:")
        for season, rnd, reason in skipped[:10]:  # Show first 10
            print(f"  • {season} R{rnd}: {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    
    if not all_feats:
        raise RuntimeError("❌ No training data collected. Check cache and data availability.")
    
    # Combine
    train_df = pd.concat(all_feats, ignore_index=True)
    train_df = train_df.sort_values(["season", "round"]).reset_index(drop=True)
    
    y = train_df["finish_position"].astype(float)
    X = train_df.drop(columns=["finish_position"])
    
    # Final validation
    if not validate_features(X, y, groups):
        raise RuntimeError("❌ Feature validation failed")
    
    return X, y, groups, race_keys, train_df


def main():
    parser = argparse.ArgumentParser(description="Train F1 Podium Prediction Model")
    parser.add_argument(
        '--seasons', 
        type=int, 
        nargs='+', 
        required=True,
        help='Seasons to train on (e.g., 2022 2023 2024)'
    )
    parser.add_argument(
        '--rounds', 
        type=int, 
        nargs='*', 
        default=None,
        help='Specific rounds (if omitted, uses all cached)'
    )
    parser.add_argument(
        '--model-out', 
        type=str, 
        default='models/f1_ranker_v1.pkl',
        help='Output model path'
    )
    parser.add_argument(
        '--no-offline',
        action='store_true',
        help='Allow online API calls (not recommended)'
    )
    parser.add_argument(
        '--min-races',
        type=int,
        default=5,
        help='Minimum races required for training'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("🏎️  F1 PODIUM PREDICTION MODEL TRAINING")
    print(f"{'='*60}")
    print(f"Seasons: {args.seasons}")
    print(f"Offline mode: {not args.no_offline}")
    print(f"Min races: {args.min_races}")
    print(f"{'='*60}\n")
    
    # Collect data
    try:
        X, y, groups, race_keys, train_df = collect_training_data(
            args.seasons, 
            args.rounds,
            offline=not args.no_offline
        )
    except Exception as e:
        print(f"\n❌ FATAL: Data collection failed: {e}")
        return 1
    
    # Check minimum data requirement
    if len(groups) < args.min_races:
        print(f"\n❌ FATAL: Insufficient data ({len(groups)} races < {args.min_races} required)")
        print("💡 Try warming cache: python scripts/warm_cache.py --seasons 2022 2023 2024")
        return 1
    
    # Train model
    print(f"\n{'='*60}")
    print("🚀 TRAINING MODEL")
    print(f"{'='*60}\n")
    
    model = F1PodiumModel()
    
    try:
        model.train(X, y, groups)
    except Exception as e:
        print(f"\n❌ FATAL: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save artifacts
    os.makedirs('models', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    
    train_df.to_csv('artifacts/training_corpus.csv', index=False)
    
    with open('artifacts/training_metadata.json', 'w') as f:
        json.dump({
            'seasons': args.seasons,
            'n_races': len(groups),
            'n_samples': len(X),
            'groups': groups,
            'race_keys': race_keys,
            'features': list(X.columns)
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Model saved: {args.model_out}")
    print(f"Training corpus: artifacts/training_corpus.csv")
    print(f"Metadata: artifacts/training_metadata.json")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())