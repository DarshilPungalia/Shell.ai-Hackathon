import pandas as pd
from xgboost import XGBRegressor, DMatrix
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import KFold
import numpy as np
from ingestion import DataFrameLoader
import optuna
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings(action='ignore')

class GradientBoostingPipeline:
    def __init__(self, regressor = None, n_trials = 20):
        self.n_trials = n_trials
        self.regressor = regressor or XGBRegressor
        if self.regressor is XGBRegressor:
            self.config = {'tree_method': 'hist',
                            'device': 'cuda',
                            'objective': 'reg:squarederror',
                            'metric':'rmse',
                            'random_state': 12
                            }  
        elif self.regressor is LGBMRegressor: 
            self.config = {'boosting_type': 'gbdt',
                            'objective': 'regression',
                            'metric': 'rmse',
                            'random_state': 12,
                            'device': 'gpu',
                            'gpu_use_dp': False,  
                            'max_bin': 255,
                            'verbose':-1
                            }
        elif self.regressor is CatBoostRegressor:
            self.config = {
                            'task_type': 'GPU',
                            'devices': '0',
                            'loss_function': 'RMSEWithUncertainty',
                            'random_seed': 12,
                            'verbose': 0
                        }
                                                                    
        self.models = []
        self.best_params = []
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        self.loader = DataFrameLoader()

        self.optimize()
        self.train()

    def objective(self, trial):
        if self.regressor is XGBRegressor:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }

        elif self.regressor is LGBMRegressor:
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

        elif self.regressor is CatBoostRegressor:
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 100.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            }

        else:
            raise ValueError("Unsupported regressor")
        
        cv_scores = []
        skf = KFold(n_splits=3, shuffle=True, random_state=42)
        

        for train_idx, val_idx in skf.split(self.x, self.y):
            x_train, x_val = self.x[train_idx], self.x[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model = self.regressor(**self.config ,**params)
            
            model.fit(x_train, y_train)
            
            if type(model) is CatBoostRegressor:
                pred = model.predict(x_val, prediction_type='RMSEWithUncertainty')[:, 0]
            else:
                pred = model.predict(x_val)
            score = mape(y_val, pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def optimize_hp(self):
        study = optuna.create_study(direction='minimize')
        print('Optimizing HyperParameter...')
        study.optimize(self.objective, n_trials=self.n_trials)
        print(f'Best trial was with MAPE: {study.best_value}')
        
        self.best_params.append(study.best_params)

    def train(self):
        pbar = tqdm(zip(self.y_split, self.best_params, self.labels), total=len(self.best_params), desc=f'Training {self.regressor.__name__}')
        for y, params, label in pbar:
            pbar.set_description(f'Training {self.regressor.__name__} for {label}')
            print(f'Shape of target: {y.shape}')
            model = self.regressor(**self.config, **params)
            model.fit(self.x, y)
            self.models.append(model)
        
        print(f'Finished Traning {len(self.models)} {self.regressor.__name__}(s)')

    def optimize(self):
        self.x, self.y_split, self.x_test = self.loader.load(split_labels=True)
        print(type(self.y_split))

        pbar = tqdm(zip(self.y_split, self.labels), total=len(self.labels), desc=f'Optimizing {self.regressor.__name__}')
        for value, label in pbar:
            pbar.set_description(f'Optimizing {self.regressor.__name__} for {label}')
            self.y = value
            self.optimize_hp()

    def get_submission(self, path):
        preds = []
        for model in tqdm(self.models, desc='Generating Predictions for each Blend', total=len(self.models)):
            if type(model) is CatBoostRegressor:
                pred = model.predict(self.x_test, prediction_type='RMSEWithUncertainty')[:, 0]
            else:
                pred = model.predict(self.x_test)
            preds.append(pred)
        
        preds = np.column_stack(preds)
        print(f'Stacked predictions shape: {preds.shape}')

        preds = self.loader.get_transformers().inverse_transform(preds)

        submission = pd.DataFrame(data=preds, columns=self.labels, index=None)
        submission['ID'] = self.loader.get_test_data()['ID']

        print(f'Saving Predictions to {path}')
        submission.to_csv(path, index=False)
        print('Predictions Saved')
    
    def get_leaves(self):
        preds = []
        for model in tqdm(self.models, desc='Extracting Leaves for each Blend', total=len(self.models)):
            if type(model) is CatBoostRegressor:
                pred = model.predict(self.x, prediction_type='LeafIndex')[:, 0]
            elif type(model) is XGBRegressor:
                booster = model.get_booster()
                dmat = DMatrix(self.x)

                pred = booster.predict(dmat, pred_leaf=True)

            elif type(model) is LGBMRegressor:
                pred = model.predict(self.x_test, pred_leaf=True)
            else:
                raise RuntimeError('Can not get Leaf Indicies')
            preds.append(pred)
    
        return preds


if __name__ == "__main__":
    model = GradientBoostingPipeline(regressor=CatBoostRegressor)
    model.get_submission(os.path.join('submissions', 'cat_dist.csv'))
