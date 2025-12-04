# src/model.py
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
    A ranking model that predicts the top 3 (podium) of an F1 race.
    Uses XGBRanker to learn *relative order* of drivers within a race.
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None

        # === ALL FEATURES FROM features.py (MUST MATCH EXACTLY) ===
        self.numeric_features = [
            'quali_rank', 'quali_delta_to_p1', 'quali_sim_fp3', 'teammate_delta_quali',
            'race_pace_fp2_best', 'pace_vs_field_fp2', 'consistency_iqr_fp2',
            'deg_best_per_lap', 'long_run_laps_best', 'teammate_delta_race',
            'race_pace_fp2_soft', 'deg_soft_per_lap', 'soft_run_laps',
            'form_avg_quali_rank_l5', 'form_avg_finish_pos_l5', 'form_avg_grid_delta_l5',
            'form_podium_rate_l5', 'form_dnf_rate_l5',
            'track_avg_finish_pos', 'track_avg_grid_delta', 'track_podium_rate', 'track_dnf_rate',
            'temp_fp2', 'track_temp_fp2', 'humidity_fp2', 'pressure_fp2', 'wind_speed_fp2',
            'is_raining', 'deg_x_temp', 'hot_weather_flag'
        ]

        self.categorical_features = ['best_compound', 'team']

        # === PREPROCESSOR: Scale + Encode ===
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ],
            remainder='drop'  # ignore any extra columns
        )

    def train(self, X: pd.DataFrame, y: pd.Series, groups: List[int]) -> None:
        """
        Train the ranking model.
        
        Args:
            X: Features (from build_features)
            y: Actual finish_position (1, 2, 3, ..., 21)
            groups: List of group sizes → [20, 20, 20, ...] one per race
        """
        print(f"Training XGBRanker on {len(X)} samples across {len(groups)} races...")

        # === 1. Preprocess ===
        X_preprocessed = self.preprocessor.fit_transform(X)
        self.feature_names = self.preprocessor.get_feature_names_out()

        # === 2. Initialize Ranker ===
        ranker = XGBRanker(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='rank:pairwise',
            random_state=42,
            n_jobs=-1
        )

        # === 3. Time-Based Cross-Validation ===
        if len(groups) > 1:
            tscv = TimeSeriesSplit(n_splits=min(5, len(groups) - 1))
            maes = []
            podium_accs = []

            cumulative_idx = np.cumsum([0] + groups[:-1])  # start of each race

            for fold, (train_idx, val_idx) in enumerate(tscv.split(cumulative_idx)):
                # Build group lists for train/val
                train_groups = groups[train_idx[0]:train_idx[-1]+1]
                val_groups = groups[val_idx[0]:val_idx[-1]+1]

                # Flatten indices
                train_flat = []
                for start, size in zip(cumulative_idx[train_idx], train_groups):
                    train_flat.extend(range(start, start + size))
                
                val_flat = []
                for start, size in zip(cumulative_idx[val_idx], val_groups):
                    val_flat.extend(range(start, start + size))

                X_train = X_preprocessed[train_flat]
                y_train = y.iloc[train_flat]
                X_val = X_preprocessed[val_flat]
                y_val = y.iloc[val_flat]

                ranker.fit(X_train, y_train, group=train_groups)
                pred = ranker.predict(X_val)

                # Convert scores to ranks
                ranked_pos = self._scores_to_ranks(pred, val_groups)
                mae = np.mean(np.abs(ranked_pos - y_val))
                podium_acc = self._podium_accuracy(ranked_pos, y_val, val_groups)
                maes.append(mae)
                podium_accs.append(podium_acc)
                print(f"  Fold {fold+1} MAE: {mae:.3f} | Podium Acc: {podium_acc:.1%}")

            print(f"CV Mean MAE: {np.mean(maes):.3f} | Mean Podium Acc: {np.mean(podium_accs):.1%}")
        else:
            print("  Only 1 race → skipping CV")

        # === 4. Final Training on All Data ===
        ranker.fit(X_preprocessed, y, group=groups)
        self.model = ranker

        # === 5. Save ===
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names
        }, "models/f1_ranker_v1.pkl")
        print("RANKER SAVED: models/f1_ranker_v1.pkl")

    def predict_podium(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict podium from features."""
        if self.model is None:
            raise ValueError("Model not trained!")

        X_preprocessed = self.preprocessor.transform(X)
        scores = self.model.predict(X_preprocessed)
        X = X.copy()
        X['predicted_score'] = scores
        X['predicted_rank'] = X['predicted_score'].rank(method='first', ascending=False)
        podium = X.sort_values('predicted_rank').head(3)[['Driver', 'predicted_rank']]
        podium['podium_step'] = [1, 2, 3]
        return podium.reset_index(drop=True)

        # === HELPERS ===
    @staticmethod
    def _scores_to_ranks(scores: np.ndarray, group_sizes: List[int]) -> np.ndarray:
        """
        Convert model scores into ranks (1 = best) within each race group.

        scores: 1D array of scores for all drivers in the validation set,
                concatenated race by race.
        group_sizes: list of group sizes for the races in this set,
                     e.g. [20, 19, 20] for three races.
        """
        ranks = np.zeros(len(scores), dtype=int)
        start = 0
        for size in group_sizes:
            end = start + size
            group_scores = scores[start:end]

            # Higher score = better rank (1)
            order = np.argsort(-group_scores)          # indices from best to worst
            ranks[start:end] = np.argsort(order) + 1   # convert to rank 1..size

            start = end
        return ranks

    @staticmethod
    def _podium_accuracy(pred_ranks: np.ndarray,
                         true_ranks: np.ndarray,
                         group_sizes: List[int]) -> float:
        """
        Podium accuracy across multiple races.

        For each race:
          - find drivers with rank <= 3 in prediction and truth
          - count intersection
          - we know there are exactly 3 podium slots per race

        Returns:
            Fraction in [0, 1], i.e. 0.67 = 67% podium accuracy.
        """
        total_correct = 0
        total_podium_slots = 0

        start = 0
        for size in group_sizes:
            end = start + size

            pr = pred_ranks[start:end]
            tr = true_ranks[start:end]

            pred_top3 = set(np.where(pr <= 3)[0])
            true_top3 = set(np.where(tr <= 3)[0])
            print("PRED TOP 3")
            print(pred_top3)
            print("TRUE TOP 3")
            print(true_top3)
            total_correct += len(pred_top3 & true_top3)
            total_podium_slots += 3  # 3 podium places per race

            start = end

        if total_podium_slots == 0:
            print("Total slots are zero")
            return 0.0

        return total_correct / total_podium_slots
