#!/usr/bin/env python3
"""
Hyperparameter tuning for F1 Podium Prediction Model.
Uses Optuna for Bayesian optimization.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRanker
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_all_races import collect_training_data
from src.model import F1PodiumModel


class HyperparameterTuner:
    def __init__(self, X, y, groups):
        self.X = X
        self.y = y
        self.groups = groups
        self.best_params = None
        self.best_score = 0
        
        # Preprocessor (fit once)
        self.model = F1PodiumModel()
        self.X_processed = self.model.preprocessor.fit_transform(X)
        self.y_values = y.to_numpy()
        self.y_relevance = -self.y_values
    
    def objective(self, trial):
        """Optuna objective function."""
        
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5),
            'objective': 'rank:pairwise',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)  # Use 3 splits for speed
        cumulative_idx = np.cumsum([0] + self.groups[:-1])
        
        podium_accs = []
        maes = []
        
        for train_race_idx, val_race_idx in tscv.split(cumulative_idx):
            # Get train/val groups
            train_groups = [self.groups[i] for i in train_race_idx]
            val_groups = [self.groups[i] for i in val_race_idx]
            
            # Flatten indices
            train_flat = []
            for i in train_race_idx:
                start = cumulative_idx[i]
                train_flat.extend(range(start, start + self.groups[i]))
            
            val_flat = []
            for i in val_race_idx:
                start = cumulative_idx[i]
                val_flat.extend(range(start, start + self.groups[i]))
            
            # Train
            ranker = XGBRanker(**params)
            ranker.fit(
                self.X_processed[train_flat], 
                self.y_relevance[train_flat], 
                group=train_groups
            )
            
            # Predict
            pred = ranker.predict(self.X_processed[val_flat])
            ranked_pos = self._scores_to_ranks(pred, val_groups)
            
            # Metrics
            mae = np.mean(np.abs(ranked_pos - self.y_values[val_flat]))
            podium_acc = self._podium_accuracy(ranked_pos, self.y_values[val_flat], val_groups)
            
            maes.append(mae)
            podium_accs.append(podium_acc)
        
        # We want to maximize podium accuracy (primary) and minimize MAE (secondary)
        mean_podium = np.mean(podium_accs)
        mean_mae = np.mean(maes)
        
        # Combined score: prioritize podium accuracy
        score = mean_podium - (mean_mae / 10)  # Scale MAE to be secondary
        
        return score
    
    @staticmethod
    def _scores_to_ranks(scores, group_sizes):
        """Convert scores to ranks within groups."""
        ranks = np.zeros(len(scores), dtype=int)
        start = 0
        for size in group_sizes:
            end = start + size
            group_scores = scores[start:end]
            order = np.argsort(-group_scores)
            ranks[start:end] = np.argsort(order) + 1
            start = end
        return ranks
    
    @staticmethod
    def _podium_accuracy(pred_ranks, true_ranks, group_sizes):
        """Calculate podium accuracy."""
        total_correct = 0
        total_slots = 0
        
        start = 0
        for size in group_sizes:
            end = start + size
            pr = pred_ranks[start:end]
            tr = true_ranks[start:end]
            pred_top3 = set(np.where(pr <= 3)[0])
            true_top3 = set(np.where(tr <= 3)[0])
            total_correct += len(pred_top3 & true_top3)
            total_slots += 3
            start = end
        
        return total_correct / total_slots if total_slots > 0 else 0.0
    
    def tune(self, n_trials=50, timeout=None):
        """Run hyperparameter tuning."""
        print(f"\n{'='*60}")
        print(f"🔍 HYPERPARAMETER TUNING")
        print(f"{'='*60}")
        print(f"Trials: {n_trials}")
        print(f"Timeout: {timeout if timeout else 'None'}")
        print(f"Data: {len(self.X)} samples, {len(self.groups)} races")
        print(f"{'='*60}\n")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(
            self.objective, 
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"\n{'='*60}")
        print(f"✅ TUNING COMPLETE")
        print(f"{'='*60}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"\nBest parameters:")
        for param, value in sorted(self.best_params.items()):
            print(f"  {param}: {value}")
        
        return self.best_params
    
    def compare_with_default(self):
        """Compare tuned params with default."""
        print(f"\n{'='*60}")
        print(f"📊 COMPARISON: Tuned vs Default")
        print(f"{'='*60}")
        
        default_params = {
            'n_estimators': 400,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'objective': 'rank:pairwise',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Evaluate default
        print("\nEvaluating default parameters...")
        default_score = self._evaluate_params(default_params)
        
        # Evaluate tuned
        print("Evaluating tuned parameters...")
        tuned_params = {**default_params, **self.best_params}
        tuned_score = self._evaluate_params(tuned_params)
        
        print(f"\nResults:")
        print(f"  Default score: {default_score:.4f}")
        print(f"  Tuned score:   {tuned_score:.4f}")
        print(f"  Improvement:   {tuned_score - default_score:+.4f} ({100*(tuned_score - default_score)/default_score:+.1f}%)")
    
    def _evaluate_params(self, params):
        """Evaluate a parameter set with full CV."""
        tscv = TimeSeriesSplit(n_splits=5)
        cumulative_idx = np.cumsum([0] + self.groups[:-1])
        
        scores = []
        
        for train_race_idx, val_race_idx in tscv.split(cumulative_idx):
            train_groups = [self.groups[i] for i in train_race_idx]
            val_groups = [self.groups[i] for i in val_race_idx]
            
            train_flat = []
            for i in train_race_idx:
                start = cumulative_idx[i]
                train_flat.extend(range(start, start + self.groups[i]))
            
            val_flat = []
            for i in val_race_idx:
                start = cumulative_idx[i]
                val_flat.extend(range(start, start + self.groups[i]))
            
            ranker = XGBRanker(**params)
            ranker.fit(
                self.X_processed[train_flat], 
                self.y_relevance[train_flat], 
                group=train_groups
            )
            
            pred = ranker.predict(self.X_processed[val_flat])
            ranked_pos = self._scores_to_ranks(pred, val_groups)
            
            mae = np.mean(np.abs(ranked_pos - self.y_values[val_flat]))
            podium_acc = self._podium_accuracy(ranked_pos, self.y_values[val_flat], val_groups)
            
            score = podium_acc - (mae / 10)
            scores.append(score)
        
        return np.mean(scores)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune F1 model hyperparameters")
    parser.add_argument('--seasons', type=int, nargs='+', default=[2022, 2023, 2024])
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, help='Timeout in seconds')
    parser.add_argument('--quick', action='store_true', help='Quick tuning (20 trials)')
    parser.add_argument('--compare', action='store_true', help='Compare with default params')
    
    args = parser.parse_args()
    
    if args.quick:
        args.trials = 20
    
    # Load data
    print("Loading training data...")
    X, y, groups, race_keys, train_df = collect_training_data(args.seasons)
    
    # Tune
    tuner = HyperparameterTuner(X, y, groups)
    best_params = tuner.tune(n_trials=args.trials, timeout=args.timeout)
    
    # Compare if requested
    if args.compare:
        tuner.compare_with_default()
    
    # Save best params
    import json
    with open('models/best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n✅ Best parameters saved to: models/best_hyperparameters.json")
    print("\nTo train with these parameters, update grok_model.py with:")
    print("="*60)
    for param, value in sorted(best_params.items()):
        print(f"  {param}={value},")
    print("="*60)


if __name__ == '__main__':
    main()