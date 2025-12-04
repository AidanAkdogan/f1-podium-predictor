"""
F1 Podium Prediction Model with robust CV and validation.
"""
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRanker
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")


class F1PodiumModel:
    """
    XGBRanker-based model for F1 podium prediction.
    Learns relative driver rankings within each race.
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None

        # Features must match build_features() output
        self.numeric_features = [
            # Qualifying (4)
            'quali_rank', 'quali_delta_to_p1', 'quali_sim_fp3', 'teammate_delta_quali',
            
            # Race Pace (5)
            'race_pace_fp2_best', 'pace_vs_field_fp2', 'consistency_iqr_fp2',
            'deg_best_per_lap', 'teammate_delta_race',
            
            # Form (5)
            'form_avg_quali_rank_l5', 'form_avg_finish_pos_l5', 'form_avg_grid_delta_l5',
            'form_podium_rate_l5', 'races_in_window',
            
            # Track History (3)
            'track_avg_finish_pos', 'track_podium_rate', 'track_avg_grid_delta',
            
            # Tire Strategy (3)
            'medium_laps_fp2', 'compounds_tried', 'longest_stint_fp2',
            
            # Weather (4)
            'temp_fp2', 'temp_deviation', 'weather_stability', 'wet_practice_laps',
            
            # Interactions (6)
            'quali_form_strength', 'track_form_synergy',
            'adaptability_score', 'teammate_dominance',
            
            # Context (1)
            'is_sprint_weekend',
            
        ]

        self.categorical_features = ['team']

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ],
            remainder='drop'
        )

    def train(self, X: pd.DataFrame, y: pd.Series, groups: List[int]) -> None:
        """
        Train the ranking model with time-series CV.
        
        Args:
            X: Features (must match self.numeric_features + self.categorical_features)
            y: Finish positions (1=best, 21=DNF)
            groups: List of group sizes [20, 19, 20, ...] one per race
        """
        print(f"\n🏋️  Training XGBRanker")
        print(f"  Samples: {len(X)}")
        print(f"  Races: {len(groups)}")
        print(f"  Features: {len(self.numeric_features) + len(self.categorical_features)}")
        
        # Validate inputs
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        
        if sum(groups) != len(X):
            raise ValueError(f"Group sum mismatch: {sum(groups)} vs {len(X)}")
        
        # Check for required columns
        missing_cols = set(self.numeric_features + self.categorical_features) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Preprocess
        print("  Preprocessing features...")
        X_preprocessed = self.preprocessor.fit_transform(X)
        self.feature_names = self.preprocessor.get_feature_names_out()
        print(f"  Preprocessed shape: {X_preprocessed.shape}")
        
        # Convert to relevance labels (higher = better)
        y_values = y.to_numpy()
        y_relevance = -y_values  # P1 → -1 (best), P20 → -20 (worst)
        
        # Initialize ranker with TUNED hyperparameters
        ranker = XGBRanker(
            colsample_bytree=0.8505915654206465,
            gamma=1.1142562304519632,
            learning_rate=0.039640708664404646,
            max_depth=9,
            min_child_weight=4,
            n_estimators=200,
            reg_alpha=1.147378935011184,
            reg_lambda=0.6434172658669481,
            subsample=0.9406069544760229,
            objective='rank:pairwise',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        # Time-series cross-validation
        if len(groups) >= 5:
            print("\n  Running time-series CV...")
            self._run_cv(X_preprocessed, y_values, y_relevance, groups, ranker)
        else:
            print(f"\n  ⚠️  Skipping CV: only {len(groups)} races (need ≥5)")
        
        # Final training on all data
        print("\n  Training final model on all data...")
        ranker.fit(X_preprocessed, y_relevance, group=groups, verbose=False)
        self.model = ranker
        
        # Feature importance
        self._print_feature_importance()
        
        # Save
        model_path = "models/f1_ranker_v1.pkl"
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names
        }, model_path)
        print(f"\n✅ Model saved: {model_path}")

    def _run_cv(self, X: np.ndarray, y_true: np.ndarray, y_relevance: np.ndarray, 
                groups: List[int], ranker: XGBRanker) -> None:
        """Run time-series cross-validation."""
        
        # Determine optimal n_splits
        max_splits = len(groups) - 1
        n_splits = min(5, max_splits)
        
        if n_splits < 2:
            print(f"    ⚠️  Cannot run CV: need ≥2 splits, have {n_splits}")
            return
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Cumulative indices for races
        cumulative_idx = np.cumsum([0] + groups[:-1])
        
        maes = []
        podium_accs = []
        
        print(f"    Folds: {n_splits}")
        
        for fold, (train_race_idx, val_race_idx) in enumerate(tscv.split(cumulative_idx)):
            # Get train/val groups
            train_groups = [groups[i] for i in train_race_idx]
            val_groups = [groups[i] for i in val_race_idx]
            
            # Flatten to sample indices
            train_flat = []
            for i in train_race_idx:
                start = cumulative_idx[i]
                train_flat.extend(range(start, start + groups[i]))
            
            val_flat = []
            for i in val_race_idx:
                start = cumulative_idx[i]
                val_flat.extend(range(start, start + groups[i]))
            
            # Split data
            X_train = X[train_flat]
            X_val = X[val_flat]
            y_train_rel = y_relevance[train_flat]
            y_val_true = y_true[val_flat]
            
            # Train on this fold
            ranker.fit(X_train, y_train_rel, group=train_groups, verbose=False)
            pred = ranker.predict(X_val)
            
            # Convert scores to ranks
            ranked_pos = self._scores_to_ranks(pred, val_groups)
            
            # Metrics
            mae = np.mean(np.abs(ranked_pos - y_val_true))
            podium_acc = self._podium_accuracy(ranked_pos, y_val_true, val_groups)
            
            maes.append(mae)
            podium_accs.append(podium_acc)
            
            print(f"    Fold {fold + 1}/{n_splits}: MAE={mae:.3f}, Podium Acc={podium_acc:.1%}")
        
        print(f"    ───────────────────────────────")
        print(f"    Mean MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f}")
        print(f"    Mean Podium Acc: {np.mean(podium_accs):.1%} ± {np.std(podium_accs):.1%}")

    def _print_feature_importance(self):
        """Print top feature importances."""
        if self.model is None:
            return
        
        try:
            importances = self.model.feature_importances_
            
            # Map back to original feature names
            feat_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\n  📊 Top 10 Features:")
            for i, row in feat_imp.head(10).iterrows():
                print(f"    {row['feature'][:40]:<40} {row['importance']:.4f}")
        except Exception as e:
            print(f"    ⚠️  Could not print importance: {e}")

    def predict_podium(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict podium from features.
        
        Returns:
            DataFrame with Driver, predicted_rank, podium_step
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        X_preprocessed = self.preprocessor.transform(X)
        scores = self.model.predict(X_preprocessed)
        
        X = X.copy()
        X['predicted_score'] = scores
        X['predicted_rank'] = X['predicted_score'].rank(method='first', ascending=False)
        
        podium = X.nsmallest(3, 'predicted_rank')[['Driver', 'predicted_rank']]
        podium['podium_step'] = [1, 2, 3]
        
        return podium.reset_index(drop=True)

    @staticmethod
    def _scores_to_ranks(scores: np.ndarray, group_sizes: List[int]) -> np.ndarray:
        """
        Convert model scores to ranks (1=best) within each race group.
        """
        ranks = np.zeros(len(scores), dtype=int)
        start = 0
        
        for size in group_sizes:
            end = start + size
            group_scores = scores[start:end]
            
            # Higher score = better rank
            order = np.argsort(-group_scores)
            ranks[start:end] = np.argsort(order) + 1
            
            start = end
        
        return ranks

    @staticmethod
    def _podium_accuracy(pred_ranks: np.ndarray, true_ranks: np.ndarray, 
                        group_sizes: List[int]) -> float:
        """
        Calculate podium accuracy across multiple races.
        
        Returns:
            Fraction of correctly predicted podium positions (0-1)
        """
        total_correct = 0
        total_slots = 0
        
        start = 0
        for size in group_sizes:
            end = start + size
            
            pr = pred_ranks[start:end]
            tr = true_ranks[start:end]
            
            # Find top 3
            pred_top3 = set(np.where(pr <= 3)[0])
            true_top3 = set(np.where(tr <= 3)[0])
            
            total_correct += len(pred_top3 & true_top3)
            total_slots += 3
            
            start = end
        
        return total_correct / total_slots if total_slots > 0 else 0.0

    def load(self, path: str = "models/f1_ranker_v1.pkl"):
        """Load a trained model."""
        data = joblib.load(path)
        self.model = data['model']
        self.preprocessor = data['preprocessor']
        self.feature_names = data['feature_names']
        print(f"✅ Model loaded from {path}")