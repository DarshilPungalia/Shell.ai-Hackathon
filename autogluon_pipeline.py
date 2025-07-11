import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from sklearn.multioutput import MultiOutputRegressor
from ingestion import DataFrameLoader
from typing import Literal
from tqdm import tqdm
import os
import warnings
import tempfile
import shutil

warnings.filterwarnings(action='ignore')

class AutoGluonPipeline:
    def __init__(self, approach: Literal['individual', 'multioutput', 'ensemble'] = 'individual'):
        self.approach = approach
        self.models = []
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        self.loader = DataFrameLoader()
        self.temp_dirs = []  # Keep track of temporary directories
        self.train()

    def train(self):
        self.x, self.y_split, self.x_test = self.loader.load(split_labels=True)
        
        # Combine all targets
        self.y = np.column_stack(self.y_split)
        print(f'Combined target shape: {self.y.shape}')
        
        # Split data
        x_train, x_val, y_train, y_val = train_test_split(
            self.x, self.y, test_size=0.2, random_state=69
        )
        
        print(f'Training set shape: {x_train.shape}, {y_train.shape}')
        print(f'Validation set shape: {x_val.shape}, {y_val.shape}')
        
        # Convert to DataFrames (AutoGluon works better with DataFrames)
        feature_names = [f'feature_{i}' for i in range(x_train.shape[1])]
        train_df = pd.DataFrame(x_train, columns=feature_names)
        val_df = pd.DataFrame(x_val, columns=feature_names)
        self.test_df = pd.DataFrame(self.x_test, columns=feature_names)
        
        if self.approach == 'individual':
            self._trainividual_models(train_df, val_df, y_train, y_val)
        elif self.approach == 'multioutput':
            self._train_multioutput_model(train_df, val_df, y_train, y_val)
        elif self.approach == 'ensemble':
            self._train_ensemble_approach(train_df, val_df, y_train, y_val)

    def _trainividual_models(self, train_df, val_df, y_train, y_val):
        """Train individual AutoGluon models for each target"""
        print('Training Individual AutoGluon models...')
        
        val_preds = []
        
        for i, label in enumerate(tqdm(self.labels, desc='Training AutoGluon models')):
            # Create temporary directory for this model
            temp_dir = tempfile.mkdtemp(prefix=f'autogluon_{label}_')
            self.temp_dirs.append(temp_dir)
            
            # Prepare data with target
            train_data = train_df.copy()
            train_data[label] = y_train[:, i]
            
            val_data = val_df.copy()
            val_data[label] = y_val[:, i]
            
            # AutoGluon predictor
            predictor = TabularPredictor(
                label=label,
                path=temp_dir,
                problem_type='regression',
                eval_metric='mean_absolute_percentage_error',
                verbosity=1
            )
            
            # Fit with optimized settings for small datasets
            predictor.fit(
                train_data,
                tuning_data=val_data,
                time_limit=300,  # 5 minutes per model
                presets='optimize_for_deployment',  # Good for small datasets
                hyperparameters={
                    'GBM': {'num_boost_round': 500, 'learning_rate': 0.05},
                    'CAT': {'iterations': 500, 'learning_rate': 0.05},
                    'XGB': {'n_estimators': 500, 'learning_rate': 0.05},
                    'RF': {'n_estimators': 200, 'max_depth': 10},
                    'XT': {'n_estimators': 200, 'max_depth': 10},
                    'KNN': {'n_neighbors': 10},
                    'LR': {},
                    'NN_TORCH': {'num_epochs': 100, 'learning_rate': 0.01}
                },
                num_cpus=4,
                num_gpus=0
            )
            
            # Get validation predictions
            val_pred = predictor.predict(val_data.drop(columns=[label]))
            val_preds.append(val_pred.values)
            
            # Store model
            self.models.append(predictor)
            
            # Print individual performance
            target_mape = mape(y_val[:, i], val_pred.values)
            target_r2 = r2(y_val[:, i], val_pred.values)
            print(f'{label}: MAPE={target_mape:.4f}, R2={target_r2:.4f}')
        
        # Calculate overall metrics
        val_preds = np.column_stack(val_preds)
        self._calculate_overall_metrics(y_val, val_preds, 'Individual AutoGluon')

    def _train_multioutput_model(self, train_df, val_df, y_train, y_val):
        """Train single AutoGluon model with MultiOutput wrapper"""
        print('Training MultiOutput AutoGluon model...')
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix='autogluon_multioutput_')
        self.temp_dirs.append(temp_dir)
        
        # Create a base predictor function
        def create_predictor():
            # We'll use a simple approach - train on concatenated data
            # This is a workaround since AutoGluon doesn't natively support multi-output
            
            # Combine all data
            all_train_data = []
            all_val_data = []
            
            for i, label in enumerate(self.labels):
                train_subset = train_df.copy()
                train_subset['target'] = y_train[:, i]
                train_subset['task_id'] = i
                all_train_data.append(train_subset)
                
                val_subset = val_df.copy()
                val_subset['target'] = y_val[:, i]
                val_subset['task_id'] = i
                all_val_data.append(val_subset)
            
            combined_train = pd.concat(all_train_data, ignoreex=True)
            combined_val = pd.concat(all_val_data, ignoreex=True)
            
            predictor = TabularPredictor(
                label='target',
                path=temp_dir,
                problem_type='regression',
                eval_metric='mean_absolute_percentage_error',
                verbosity=1
            )
            
            predictor.fit(
                combined_train,
                tuning_data=combined_val,
                time_limit=600,  # 10 minutes total
                presets='medium_quality',
                hyperparameters={
                    'GBM': {'num_boost_round': 300},
                    'CAT': {'iterations': 300},
                    'XGB': {'n_estimators': 300},
                    'RF': {'n_estimators': 150},
                    'NN_TORCH': {'num_epochs': 50}
                }
            )
            
            return predictor
        
        self.model = create_predictor()
        
        # Get validation predictions
        val_preds = []
        for i in range(len(self.labels)):
            val_subset = val_df.copy()
            val_subset['task_id'] = i
            pred = self.model.predict(val_subset)
            val_preds.append(pred.values)
        
        val_preds = np.column_stack(val_preds)
        self._calculate_overall_metrics(y_val, val_preds, 'MultiOutput AutoGluon')

    def _train_ensemble_approach(self, train_df, val_df, y_train, y_val):
        """Train ensemble of different AutoGluon configurations"""
        print('Training Ensemble AutoGluon approach...')
        
        # Different configurations
        configs = [
            {'preset': 'medium_quality', 'time_limit': 200},
            {'preset': 'good_quality', 'time_limit': 300},
            {'preset': 'optimize_for_deployment', 'time_limit': 150}
        ]
        
        ensemble_preds = []
        
        for config_idx, config in enumerate(configs):
            print(f'Training configuration {config_idx + 1}/{len(configs)}...')
            config_preds = []
            
            for i, label in enumerate(self.labels):
                temp_dir = tempfile.mkdtemp(prefix=f'autogluon_ensemble_{config_idx}_{label}_')
                self.temp_dirs.append(temp_dir)
                
                train_data = train_df.copy()
                train_data[label] = y_train[:, i]
                
                val_data = val_df.copy()
                val_data[label] = y_val[:, i]
                
                predictor = TabularPredictor(
                    label=label,
                    path=temp_dir,
                    problem_type='regression',
                    eval_metric='mean_absolute_percentage_error',
                    verbosity=0
                )
                
                predictor.fit(
                    train_data,
                    tuning_data=val_data,
                    presets=config['preset'],
                    time_limit=config['time_limit'],
                    num_cpus=2,
                    num_gpus=0
                )
                
                val_pred = predictor.predict(val_data.drop(columns=[label]))
                config_preds.append(val_pred.values)
            
            ensemble_preds.append(np.column_stack(config_preds))
        
        # Average ensemble predictions
        val_preds = np.mean(ensemble_preds, axis=0)
        self.ensemble_preds = ensemble_preds  # Store for later use
        
        self._calculate_overall_metrics(y_val, val_preds, 'Ensemble AutoGluon')

    def _calculate_overall_metrics(self, y_true, y_pred, approach_name):
        """Calculate and print overall metrics"""
        overall_mape = 0
        overall_r2 = 0
        
        print(f"\n=== {approach_name} Results ===")
        for i, label in enumerate(self.labels):
            target_mape = mape(y_true[:, i], y_pred[:, i])
            target_r2 = r2(y_true[:, i], y_pred[:, i])
            
            print(f'{label}: MAPE={target_mape:.4f}, R2={target_r2:.4f}')
            overall_mape += target_mape
            overall_r2 += target_r2
        
        avg_mape = overall_mape / len(self.labels)
        avg_r2 = overall_r2 / len(self.labels)
        
        print(f"\n=== Overall Performance ===")
        print(f'Average MAPE: {avg_mape:.4f}')
        print(f'Average R2: {avg_r2:.4f}')

    def get_submission(self, path):
        print(f'Generating AutoGluon predictions ({self.approach} approach)...')
        
        if self.approach == 'individual':
            preds = []
            for i, model in enumerate(tqdm(self.models, desc='Predicting with individual models')):
                pred = model.predict(self.test_df)
                preds.append(pred.values)
            preds = np.column_stack(preds)
            
        elif self.approach == 'multioutput':
            preds = []
            for i in range(len(self.labels)):
                test_subset = self.test_df.copy()
                test_subset['task_id'] = i
                pred = self.model.predict(test_subset)
                preds.append(pred.values)
            preds = np.column_stack(preds)
            
        elif self.approach == 'ensemble':
            # This would require storing all models - simplified for now
            preds = []
            for i, model in enumerate(tqdm(self.models[-len(self.labels):], desc='Predicting with last config')):
                pred = model.predict(self.test_df)
                preds.append(pred.values)
            preds = np.column_stack(preds)
        
        print(f'Test predictions shape: {preds.shape}')
        
        # Inverse transform
        preds = self.loader.get_transformers().inverse_transform(preds)
        
        # Create submission
        submission = pd.DataFrame(data=range(1, 501), columns=['ID'], index=None)
        submission[self.labels] = preds
        
        print(f'Saving predictions to {path}')
        submission.to_csv(path, index=False)
        print('Predictions saved!')

    def cleanup(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        print(f'Cleaned up {len(self.temp_dirs)} temporary directories')

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()


if __name__ == "__main__":
    # Try Individual AutoGluon models (recommended for correlated outputs)
    print("=== Individual AutoGluon Models ===")
    model = AutoGluonPipeline(approach='multioutput')
    model.get_submission(os.path.join('submissions', 'autogluon_multi.csv'))
    
    # Optional: Try MultiOutput approach
    # print("\n=== MultiOutput AutoGluon ===")
    # model_mo = AutoGluonPipeline(approach='multioutput')
    # model_mo.get_submission(os.path.join('submissions', 'autogluon_multioutput.csv'))
    
    # Clean up
    model.cleanup()