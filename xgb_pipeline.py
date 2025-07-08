import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import KFold
import numpy as np
from ingestion import DataFrameLoader
import optuna
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings(action='ignore')

class XGBPipeline:
    def __init__(self):
        self.config = {'tree_method': 'hist',
                        'device': 'cuda',
                        'objective': 'reg:squarederror',
                        'metric':'rmse',
                        'random_state': 12
                        }
        self.models = []
        self.best_params = []
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        self.loader = DataFrameLoader()

        self.optimize()
        self.train()

    def objective(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
        }
        
        cv_scores = []
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        

        for train_idx, val_idx in skf.split(self.x, self.y):
            x_train, x_val = self.x[train_idx], self.x[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model = XGBRegressor(**self.config ,**params)
            model.fit(x_train, y_train, verbose=False)
            
            pred = model.predict(x_val)
            score = mape(y_val, pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def optimize_hp(self, n_trials=20):
        
        study = optuna.create_study(direction='minimize')
        print('Optimizing HyperParameter...')
        study.optimize(self.objective, n_trials=n_trials)
        print(f'Best trial was with Params: {study.best_params}')
        
        self.best_params.append(study.best_params)

    def train(self):
        pbar = tqdm(zip(self.y_split, self.best_params, self.labels), total=len(self.best_params), desc='Training XGB')
        for y, params, label in pbar:
            pbar.set_description(f'Training XGB for {label}')
            print(f'Shape of target: {y.shape}')
            model = XGBRegressor(**self.config, **params)
            model.fit(self.x, y)
            self.models.append(model)
        
        print(f'Finished Traning {len(self.best_params)} XGBoost(s)')

    def optimize(self):
        self.x, self.y_split, self.x_test = self.loader.load(split_labels=True)
        print(type(self.y_split))

        pbar = tqdm(zip(self.y_split, self.labels), total=len(self.labels), desc='Optimizing XGB')
        for value, label in pbar:
            pbar.set_description(f'Optimizing XGB for {label}')
            self.y = value
            self.optimize_hp()

    def get_submission(self, path):
        preds = []
        for model in tqdm(self.models, desc='Generating Predictions for each Blend', total=len(self.models)):
            pred = model.predict(self.x_test)
            print(f'{type(pred)} {pred.shape}')
            preds.append(pred)
        
        preds = np.column_stack(preds)
        print(f'Stacked predictions shape: {preds.shape}')

        preds = self.loader.get_transformers().inverse_transform(preds)

        submission = pd.DataFrame(data=preds, columns=self.labels, index=None)
        submission['ID'] = self.loader.get_test_data()['ID']

        print(f'Saving Predictions to {path}')
        submission.to_csv(path, index=False)
        print('Predictions Saved')


if __name__ == "__main__":
    model = XGBPipeline()
    model.get_submission(os.path.join('submissions', 'xgb.csv'))
