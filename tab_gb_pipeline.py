import pandas as pd
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from ingestion import DataFrameLoader
from huggingface_hub import hf_hub_download
from typing import Literal
from tqdm import tqdm
import os
import warnings
from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import optuna

warnings.filterwarnings(action='ignore')

class TabPFNStackingPipeline:
    def __init__(self, output_type: Literal['mean', 'median', 'mode'] = None, meta_model = None, combine_features: bool = False, n_trials:int=20):
        self.n_trials = n_trials
        self.meta_model = meta_model or XGBRegressor
        self.combine_features = combine_features
        self.output_type = output_type or 'mean'
        self.tabpfn_models = []
        self.meta_models = []
        self.config = self.get_config(self.meta_model) 
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        self.loader = DataFrameLoader()
        
        print('Downloading TabPFN Model from HuggingFace...')
        self.path = hf_hub_download(repo_id='Prior-Labs/TabPFN-v2-reg',
                               filename='tabpfn-v2-regressor.ckpt',
                               cache_dir=r'models')
        print(f'TabPFN Model saved at {self.path}')
        self.regressor = TabPFNRegressor
        self.train()

    @staticmethod
    def get_config(model):
        if model is XGBRegressor:
            return {'tree_method': 'hist',
                    'device': 'cuda',
                    'objective': 'reg:squarederror',
                    'metric':'rmse',
                    'random_state': 12
                    }  
        elif model is LGBMRegressor: 
            return{'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    'random_state': 12,
                    'device': 'gpu',
                    'gpu_use_dp': False,  
                    'max_bin': 255,
                    'verbose':-1
                    }
        elif model is CatBoostRegressor:
            return {
                    'task_type': 'GPU',
                    'devices': '0',
                    'loss_function': 'RMSE',
                    'random_seed': 12,
                    'verbose': 0
                    }    
        elif model is HistGradientBoostingRegressor:
            return {
                    'random_state': 12,
                    'loss': 'squared_error',
                    'verbose': 0,
                    'early_stopping': True,
                    'scoring': 'neg_root_mean_squared_error'
                }    
        elif model is RandomForestRegressor:
            return {
                    'random_state': 12,
                    'n_jobs': -1,
                    'verbose': 0
                }
        else:
            raise ValueError(f'No meta model of type {model}')

    def get_quantile_features(self, model, X):
        """Extract quantile features from TabPFN model"""
        # Get all predictions at once using output_type='full'
        full_preds = model.predict(X, output_type='full')
        
        quantile_features = []
        
        # Add quantile predictions (0.1 to 0.9)
        for q_pred in full_preds['quantiles']:
            quantile_features.append(q_pred.reshape(-1, 1))
        
        # Add mean, median, mode predictions
        quantile_features.append(full_preds['mean'].reshape(-1, 1))
        quantile_features.append(full_preds['median'].reshape(-1, 1))
        quantile_features.append(full_preds['mode'].reshape(-1, 1))
        
        return np.hstack(quantile_features)
    
    @staticmethod
    def get_params_for_trial(model, trial):
        if model is XGBRegressor:
            return {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }

        elif model is LGBMRegressor:
            return {
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),  
                    'n_estimators': trial.suggest_int('n_estimators', 300, 1500),  
                    'max_depth': trial.suggest_int('max_depth', 3, 7),  
                    'num_leaves': trial.suggest_int('num_leaves', 4, 48), 
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),  
                    'subsample': trial.suggest_float('subsample', 0.5, 0.9),  
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),  
                    'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0), 
                    'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0)  
                }

        elif model is CatBoostRegressor:
            return {
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
        
        elif model is HistGradientBoostingRegressor:
            return {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_iter': trial.suggest_int('max_iter', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 16),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
                    'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 100.0, log=True)
                }
        
        elif model is RandomForestRegressor:
            return {
                    'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                    'max_depth': trial.suggest_int('max_depth', 4, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 50, 1000),
                    'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error']),
                }

        else:
            raise ValueError("Unsupported regressor")

    def objective(self, trial):
        params = self.get_params_for_trial(self.meta_model, trial)
        
        cv_scores = []
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        

        for train_idx, val_idx in skf.split(self.trainData, self.y_train):
            x_train, x_val = self.trainData[train_idx], self.trainData[val_idx]
            y_train, y_val = self.y_train[train_idx], self.y_train[val_idx]
            
            model = self.meta_model(**self.config ,**params)
            
            model.fit(x_train, y_train)
            
            pred = model.predict(x_val)
            score = mape(y_val, pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def optimize(self):
        study = optuna.create_study(direction='minimize')
        print('Optimizing HyperParameter...')
        study.optimize(self.objective, n_trials=self.n_trials)
        print(f'Best trial was with MAPE: {study.best_value}')
        
        return study.best_params

    def train(self):
        self.x, self.y_split, self.x_test = self.loader.load(split_labels=True, add_features=True)
        
        print(f"Training pipeline with TabPFN full predictions")
        print(f"This will create 12 features per target (9 quantiles + mean + median + mode)")
        
        pbar = tqdm(zip(self.y_split, self.labels), total=len(self.y_split), desc='Training TabPFN + Meta Model', dynamic_ncols=True)
        
        for y, label in pbar:
            pbar.set_description(f'Training TabPFN + Meta Model for {label}')

            x_train, x_val, self.y_train, y_val = train_test_split(self.x, y, test_size=0.2, random_state=69)

            # Step 1: Train TabPFN model
            pbar.write("Step 1: Training TabPFN model...")
            tabpfn_model = self.regressor(
                device='auto', 
                model_path=self.path
            )
            tabpfn_model.fit(x_train, self.y_train)
            self.tabpfn_models.append(tabpfn_model)

            # Step 2: Generate quantile features for training meta
            pbar.write("Step 2: Generating quantile features...")
            train_quantile_features = self.get_quantile_features(tabpfn_model, x_train)
            val_quantile_features = self.get_quantile_features(tabpfn_model, x_val)

            self.trainData = train_quantile_features if not self.combine_features else np.hstack([x_train, train_quantile_features])
            valData = val_quantile_features if not self.combine_features else np.hstack([x_val, val_quantile_features])
            
            pbar.write(f'Train quantile features shape: {self.trainData.shape}')
            pbar.write(f'Val quantile features shape: {valData.shape}')

            pbar.write("Step 3: Optimizing Meta Model model...")
            params = self.optimize()

            pbar.write("Step 4: Training Meta Model model...")
            model = self.meta_model(
                **self.config,
                **params
            )

            model.fit(
                self.trainData, self.y_train
            )
                
            pbar.write("Step 5: Validating combined model...")
            val_preds = model.predict(valData)
                        
            val_mape = mape(y_val, val_preds)
            val_r2 = r2(y_val, val_preds)
            
            pbar.write(f'Combined Model MAPE for {label}: {val_mape:.4f}')
            pbar.write(f'Combined Model R2 for {label}: {val_r2:.4f}')
            
            # Compare with TabPFN-only performance
            tabpfn_val_preds = tabpfn_model.predict(x_val, output_type=self.output_type)
            tabpfn_mape = mape(y_val, tabpfn_val_preds)
            tabpfn_r2 = r2(y_val, tabpfn_val_preds)
            
            pbar.write(f'TabPFN-only MAPE for {label}: {tabpfn_mape:.4f}')
            pbar.write(f'TabPFN-only R2 for {label}: {tabpfn_r2:.4f}')
            
            comparison_mape = ((tabpfn_mape - val_mape) / tabpfn_mape) * 100
            comparison_r2 = ((val_r2 - tabpfn_r2) / abs(tabpfn_r2)) * 100

            if comparison_mape > 0:
                pbar.write(f'Would use Meta Model for predictions for {label}')
                self.meta_models.append(model)
            else:
                pbar.write(f'Would use TabPFN for predictions for {label}')
                self.meta_models.append(tabpfn_model)
            
            pbar.write(f'MAPE comparison: {comparison_mape:.2f}%')
            pbar.write(f'R2 comparison: {comparison_r2:.2f}%')
        
        print(f'\nFinished Training {len(self.tabpfn_models)} TabPFN models and {len(self.meta_models)} Meta Model models')

    def get_submission(self, path, only_preds: bool = False):
        print(f'Generating Predictions using TabPFN quantiles + Meta Model...')
        
        preds = []
        
        pbar = tqdm(zip(self.tabpfn_models, self.meta_models, self.labels), 
                   total=len(self.tabpfn_models), 
                   desc='Generating final predictions', dynamic_ncols=True)
        
        for tabpfn_model, model, label in pbar:
            pbar.set_description(f'Predicting {label}')

            if isinstance(model, TabPFNRegressor):
                pred = model.predict(self.x_test, output_type=self.output_type)
                  
            else:
                # Generate quantile features for test set
                test_quantile_features = self.get_quantile_features(tabpfn_model, self.x_test)
                testData = np.hstack([self.x_test, test_quantile_features]) if self.combine_features else test_quantile_features
                pred = model.predict(testData) 

            preds.append(pred)
        
        preds = np.column_stack(preds)
        print(f'Stacked predictions shape: {preds.shape}')

        # Inverse transform predictions
        preds = self.loader.get_transformers().inverse_transform(preds)

        if only_preds:
            return preds

        # Create submission dataframe
        submission = pd.DataFrame(data=range(1, 501), columns=['ID'], index=None)
        submission[self.labels] = preds

        print(f'Saving Predictions to {path}')
        submission.to_csv(path, index=False)
        print('Predictions Saved')


if __name__ == "__main__":   
    print("=== Training Individual TabPFN + XGB Models with Full features===")
    model1 = TabPFNStackingPipeline(meta_model=RandomForestRegressor, n_trials=20, combine_features=True)
    model1.get_submission(os.path.join('submissions', 'tabpfn_quantile_hist_best_model_for_prediction.csv'))