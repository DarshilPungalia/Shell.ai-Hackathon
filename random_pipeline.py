import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import KFold
import numpy as np
from ingestion import DataFrameLoader
import optuna
import os
import warnings

warnings.filterwarnings(action='ignore')

class ForestPipeline:
    def __init__(self):
        self.config = {'random_state': 12, 'n_jobs': -1}
        self.model = None
        self.best_params = None
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        self.loader = DataFrameLoader()

        self.optimize()
        self.train()

    def objective(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        }
        
        cv_scores = []
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        

        for train_idx, val_idx in skf.split(self.x, self.y):
            x_train, x_val = self.x[train_idx], self.x[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model = RandomForestRegressor(**self.config ,**params)
            model.fit(x_train, y_train)
            
            pred = model.predict(x_val)
            score = mape(y_val, pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def optimize_hp(self, n_trials=50):
        study = optuna.create_study(direction='minimize')
        print('Optimizing HyperParameter...')
        study.optimize(self.objective, n_trials=n_trials)
        print(f'Best trial was with Params: {study.best_params}')
        
        self.best_params = study.best_params

    def train(self):
        print('Started Model Training...')
        model = RandomForestRegressor(**self.config, **self.best_params)
        model.fit(self.x, self.y)
        self.model = model
        
        print(f'Finished Traning Random Forest')

    def optimize(self):
        self.x, self.y, self.x_test = self.loader.load()
        print(type(self.y))
        self.optimize_hp()

    def get_submission(self, path):
        print('Generating Predictions...')
        pred = self.model.predict(self.x_test)
        pred = self.loader.get_transformers().inverse_transform(pred)

        submission = pd.DataFrame(data=range(1,501), columns=['ID'], index=None)
        submission[self.labels] = pred

        print(f'Saving Predictions to {path}')
        submission.to_csv(path, index=False)
        print('Predictions Saved')


if __name__ == "__main__":
    model = ForestPipeline()
    model.get_submission(os.path.join('submissions', 'forest.csv'))