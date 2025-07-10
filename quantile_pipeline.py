import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from ingestion import DataFrameLoader
import optuna
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')


class QuantileLightGBMPipeline:
    def __init__(self):
        self.quantiles = [round(q, 1) for q in np.arange(0.1, 1.0, 0.1)]
        self.config_base = {
            'boosting_type': 'gbdt',
            'random_state': 12,
            'device': 'gpu',
            'gpu_use_dp': False,
            'max_bin': 255,
            'verbose': -1
        }
        self.models = {q: [] for q in self.quantiles}
        self.best_params = []
        self.labels = [f'BlendProperty{i}' for i in range(1, 11)]
        self.loader = DataFrameLoader()

        self.optimize()
        self.train()

    def objective(self, trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'min_child_samples': trial.suggest_int('min_child_samples', 2, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0)
        }

        cv_scores = []
        skf = KFold(n_splits=3, shuffle=True, random_state=42)

        for train_idx, val_idx in skf.split(self.x, self.y):
            x_train, x_val = self.x[train_idx], self.x[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            model = LGBMRegressor(objective='quantile', alpha=0.5, **self.config_base, **params)
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            score = mape(y_val, pred)
            cv_scores.append(score)

        return np.mean(cv_scores)

    def optimize_hp(self, n_trials=20):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        self.best_params.append(study.best_params)

    def optimize(self):
        self.x, self.y_split, self.x_test = self.loader.load(split_labels=True)
        print(f"Optimizing for {len(self.labels)} targets")

        pbar = tqdm(zip(self.y_split, self.labels), total=len(self.labels), desc='Optimizing')
        for y, label in pbar:
            pbar.set_description(f"\nOptimizing for {label}")
            self.y = y
            self.optimize_hp()

    def train(self):
        print("\nTraining Quantile Models")
        for y, params, label in tqdm(zip(self.y_split, self.best_params, self.labels), total=len(self.labels), desc="Training"):
            print(f"\nTraining models for {label}")
            for q in self.quantiles:
                q_model = LGBMRegressor(objective='quantile', alpha=q, **self.config_base, **params)
                q_model.fit(self.x, y)
                self.models[q].append(q_model)
        print(f"\nFinished Training {len(self.quantiles)} quantile models per target")

    def get_submission(self, path):
        print("\nGenerating submission...")
        all_preds = []
        for i in range(len(self.labels)):
            target_preds = []
            for q in self.quantiles:
                model = self.models[q][i]
                pred = model.predict(self.x_test)
                target_preds.append(pred)

            avg_pred = np.mean(np.column_stack(target_preds), axis=1)
            all_preds.append(avg_pred)

        all_preds = np.column_stack(all_preds)
        all_preds = self.loader.get_transformers().inverse_transform(all_preds)

        submission = pd.DataFrame(all_preds, columns=self.labels)
        submission['ID'] = self.loader.get_test_data()['ID']

        print(f"Saving predictions to {path}")
        submission.to_csv(path, index=False)
        print("Submission saved.")


if __name__ == "__main__":
    pipeline = QuantileLightGBMPipeline()
    pipeline.get_submission(os.path.join('submissions', 'lgb_quantile.csv'))
