"""
ML Challenge 2026: Business Entity Resolution
Model Training & Inference Module
"""

import numpy as np
import lightgbm as lgb
from collections import defaultdict
from features import FEATURE_NAMES

class EntityResolutionModel:
    def __init__(self, threshold: float = 0.68, singleton_threshold: float = 0.70):
        self.threshold = threshold
        self.singleton_threshold = singleton_threshold
        self.booster = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, num_trees: int = 200):
        """Train LightGBM binary classifier on candidate pairs."""
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.08,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 1,
            'min_child_samples': 20,
            'n_jobs': 8,
            'verbose': -1
        }
        self.booster = lgb.train(params, dtrain, num_boost_round=num_trees)

    def optimize_threshold(self, X_val: np.ndarray, y_val: np.ndarray, groups_val: np.ndarray, actual_counts: dict[int, int]) -> float:
        """Perform fine-grained grid search for optimal Macro F0.5 threshold."""
        val_preds = self.booster.predict(X_val)
        unique_groups = np.unique(groups_val)

        best_score = -1.0
        best_thresh = 0.65

        for thresh in np.arange(0.50, 0.90, 0.02):
            f05_scores = []
            for g in unique_groups:
                g_mask = (groups_val == g)
                g_preds = val_preds[g_mask]
                g_labels = y_val[g_mask]

                pred_pos = (g_preds >= thresh)
                tp = np.sum(pred_pos & (g_labels == 1))
                fp = np.sum(pred_pos & (g_labels == 0))

                true_count = actual_counts.get(g, 0)
                if true_count == 0:
                    f05_scores.append(1.0 if np.sum(pred_pos) == 0 else 0.0)
                else:
                    if np.sum(pred_pos) == 0:
                        f05_scores.append(0.0)
                    else:
                        prec = tp / max(1, (tp + fp))
                        rec = tp / max(1, true_count)
                        denom = 0.25 * prec + rec
                        f05 = (1.25 * prec * rec / denom) if denom > 0 else 0.0
                        f05_scores.append(f05)

            mean_f05 = float(np.mean(f05_scores))
            if mean_f05 > best_score:
                best_score = mean_f05
                best_thresh = float(thresh)

        print(f'Optimized Macro F_0.5 Threshold: {best_thresh:.2f} (Val F_0.5 = {best_score:.5f})')
        self.threshold = best_thresh
        self.singleton_threshold = best_thresh
        return best_thresh

    def predict_pairs(self, X_pairs: np.ndarray) -> np.ndarray:
        """Predict match probability for feature array."""
        if len(X_pairs) == 0:
            return np.array([])
        return self.booster.predict(X_pairs, num_threads=8)