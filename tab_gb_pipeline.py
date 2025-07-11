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
import optuna

warnings.filterwarnings(action='ignore')

class TabPFNQuantileGBPipeline:
    def __init__(self, output_type: Literal['mean', 'median', 'mode'] = None, meta_model = None):
        self.meta_model = meta_model or XGBRegressor
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
                    'loss_function': 'RMSEWithUncertainty',
                    'random_seed': 12,
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
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'num_leaves': trial.suggest_int('num_leaves', 8, 64),
                'min_child_samples': trial.suggest_int('min_child_samples', 2, 15),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0)
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

        else:
            raise ValueError("Unsupported regressor")

    def objective(self, trial):
        params = self.get_params_for_trial(self.meta_model, trial)
        
        cv_scores = []
        skf = KFold(n_splits=3, shuffle=True, random_state=42)
        

        for train_idx, val_idx in skf.split(self.train_quantile_features, self.y_train):
            x_train, x_val = self.train_quantile_features[train_idx], self.train_quantile_features[val_idx]
            y_train, y_val = self.y_train[train_idx], self.y_train[val_idx]
            
            model = self.meta_model(**self.config ,**params)
            
            model.fit(x_train, y_train)
            
            pred = model.predict(x_val)
            score = mape(y_val, pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def optimize(self, n_trials=50):
        
        study = optuna.create_study(direction='minimize')
        print('Optimizing HyperParameter...')
        study.optimize(self.objective, n_trials=n_trials)
        print(f'Best trial was with MAPE: {study.best_value}')
        
        return study.best_params

    def train(self):
        self.x, self.y_split, self.x_test = self.loader.load(split_labels=True)
        
        print(f"Training pipeline with TabPFN full predictions")
        print(f"This will create 12 features per target (9 quantiles + mean + median + mode)")
        
        pbar = tqdm(zip(self.y_split, self.labels), total=len(self.y_split), desc='Training TabPFN + Meta Model')
        
        for y, label in pbar:
            pbar.set_description(f'Training TabPFN + Meta Model for {label}')
            print(f'\n=== Training for {label} ===')
            print(f'Shape of target: {y.shape}')

            x_train, x_val, self.y_train, y_val = train_test_split(self.x, y, test_size=0.2, random_state=69)

            print(f'Shape of x_train: {x_train.shape}')
            print(f'Shape of x_val: {x_val.shape}')
            print(f'Shape of y_train: {self.y_train.shape}')
            print(f'Shape of y_val: {y_val.shape}')

            # Step 1: Train TabPFN model
            print("Step 1: Training TabPFN model...")
            tabpfn_model = self.regressor(
                device='auto', 
                model_path=r'models\models--Prior-Labs--TabPFN-v2-reg\snapshots\213f8e38ec399a2a385fa46cab6f22b95cd90de8\tabpfn-v2-regressor.ckpt'
            )
            tabpfn_model.fit(x_train, self.y_train)
            self.tabpfn_models.append(tabpfn_model)

            # Step 2: Generate quantile features for training meta
            print("Step 2: Generating quantile features...")
            self.train_quantile_features = self.get_quantile_features(tabpfn_model, x_train)
            val_quantile_features = self.get_quantile_features(tabpfn_model, x_val)
            
            print(f'Train quantile features shape: {self.train_quantile_features.shape}')
            print(f'Val quantile features shape: {val_quantile_features.shape}')

            # Step 3: Optimize meta model
            print("Step 3: Optimizing Meta Model model...")
            params = self.optimize()

            # Step 4: Train meta model
            print("Step 4: Training Meta Model model...")
            model = self.meta_model(
                **self.config,
                **params
            )
            
            model.fit(
                self.train_quantile_features, self.y_train,
            )
            self.meta_models.append(model)

            # Step 5: Validate the combined model
            print("Step 4: Validating combined model...")
            val_preds = model.predict(val_quantile_features)
            
            val_mape = mape(y_val, val_preds)
            val_r2 = r2(y_val, val_preds)
            
            print(f'Combined Model MAPE for {label}: {val_mape:.4f}')
            print(f'Combined Model R2 for {label}: {val_r2:.4f}')
            
            # Compare with TabPFN-only performance
            tabpfn_val_preds = tabpfn_model.predict(x_val, output_type=self.output_type)
            tabpfn_mape = mape(y_val, tabpfn_val_preds)
            tabpfn_r2 = r2(y_val, tabpfn_val_preds)
            
            print(f'TabPFN-only MAPE for {label}: {tabpfn_mape:.4f}')
            print(f'TabPFN-only R2 for {label}: {tabpfn_r2:.4f}')
            
            comparison_mape = ((tabpfn_mape - val_mape) / tabpfn_mape) * 100
            comparison_r2 = ((val_r2 - tabpfn_r2) / abs(tabpfn_r2)) * 100
            
            print(f'MAPE comparison: {comparison_mape:.2f}%')
            print(f'R2 comparison: {comparison_r2:.2f}%')
        
        print(f'\nFinished Training {len(self.tabpfn_models)} TabPFN models and {len(self.meta_models)} Meta Model models')

    def get_submission(self, path):
        print(f'Generating Predictions using TabPFN quantiles + Meta Model...')
        
        preds = []
        
        pbar = tqdm(zip(self.tabpfn_models, self.meta_models, self.labels), 
                   total=len(self.tabpfn_models), 
                   desc='Generating final predictions')
        
        for tabpfn_model, model, label in pbar:
            pbar.set_description(f'Predicting {label}')
            
            # Generate quantile features for test set
            test_quantile_features = self.get_quantile_features(tabpfn_model, self.x_test)
            
            # Make final prediction with Meta Model
            pred = model.predict(test_quantile_features)
            preds.append(pred)
        
        preds = np.column_stack(preds)
        print(f'Stacked predictions shape: {preds.shape}')

        # Inverse transform predictions
        preds = self.loader.get_transformers().inverse_transform(preds)

        # Create submission dataframe
        submission = pd.DataFrame(data=range(1, 501), columns=['ID'], index=None)
        submission[self.labels] = preds

        print(f'Saving Predictions to {path}')
        submission.to_csv(path, index=False)
        print('Predictions Saved')


if __name__ == "__main__":    
    print("=== Training Individual TabPFN + Meta Models ===")
    model1 = TabPFNQuantileGBPipeline()
    model1.get_submission(os.path.join('submissions', 'tabpfn_quantile_xgb.csv'))