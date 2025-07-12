import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error as mape
from ingestion import DataFrameLoader
import optuna
import os
import warnings

warnings.filterwarnings(action='ignore')

class MultiTaskNN:
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def build_model(self, trial):
        activation = trial.suggest_categorical('act_func', ['relu', 'tanh', 'gelu'])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        inputs = tf.keras.layers.Input(shape=(self.input_dim,))

        num_hidden = trial.suggest_int('num_hidden', 2, 5)
        x = inputs
        for hidden in range(num_hidden):
            dense_units = trial.suggest_int(f'dense_units_{hidden+1}', 32, 256)
            x = tf.keras.layers.Dense(dense_units, activation=activation)(x)

        # Create 10 separate output heads for multi-task learning
        outputs = [tf.keras.layers.Dense(1, name=f"BlendProperty{i+1}")(x) for i in range(10)]

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Use MAE instead of MAPE for loss function to avoid division by zero issues
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            loss='mae'        
            )
        
        return model
            
    def objective(self, trial):
        cv_scores = []
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(self.x, self.y):
            x_train, x_val = self.x[train_idx], self.x[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            # Split y into list of arrays for multi-task outputs
            y_train_list = [y_train[:, i] for i in range(y_train.shape[1])]
            y_val_list = [y_val[:, i] for i in range(y_val.shape[1])]
            
            model = self.build_model(trial)
            
            # Add early stopping to prevent overfitting
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True
            )
            
            model.fit(
                x_train, 
                y_train_list, 
                validation_data=(x_val, y_val_list),
                epochs=50,  # Increased epochs with early stopping
                batch_size=80,
                callbacks=[early_stopping],
                verbose=0
            )

            pred = model.predict(x_val, verbose=0)
            
            pred_array = np.column_stack(pred)
            
            task_scores = []
            for i in range(y_val.shape[1]):
                task_score = mape(y_val[:, i], pred_array[:, i])
                task_scores.append(task_score)
            
            score = np.mean(task_scores)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def optimize_model(self, x, y, n_trials: int = 20):
        self.x, self.y = x, y

        study = optuna.create_study(direction='minimize')
        print('Optimizing HyperParameters...')
        study.optimize(self.objective, n_trials=n_trials)
        print(f'Best trial was with MAPE: {study.best_value:.4f}')

        return self.build_model(study.best_trial)


class MultiTaskPipeline:
    def __init__(self):
        self.labels = [f"BlendProperty{i+1}" for i in range(10)]
        self.loader = DataFrameLoader()
        self.model = None
        self.build()

    def build(self):
        self.x, self.y, self.x_test = self.loader.load()

        print(f"Training Multi-Task NN with input shape: {self.x.shape}")
        nn = MultiTaskNN(input_dim=self.x.shape[1])
        self.model = nn.optimize_model(self.x, self.y)
        self.train()

    def train(self):
        # Split y into list of arrays for multi-task outputs
        y_train_list = [self.y[:, i] for i in range(self.y.shape[1])]
        
        # Add early stopping for final training
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', 
            patience=5, 
            restore_best_weights=True
        )
        
        print("Training final model...")
        self.model.fit(
            self.x,
            y_train_list,
            epochs=100,  # More epochs with early stopping
            batch_size=100,
            callbacks=[early_stopping],
            verbose=1
        )

    def get_submission(self, path):
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
            
        print("Generating predictions...")
        preds = self.model.predict(self.x_test, verbose=0)
        
        # Convert predictions to numpy array
        preds_array = np.column_stack(preds)
        
        # Inverse transform predictions if transformer is available
        try:
            preds_array = self.loader.get_transformers().inverse_transform(preds_array)
        except AttributeError:
            print("Warning: No transformer found, using raw predictions")
        
        submission = pd.DataFrame(data=preds_array, columns=self.labels)
        
        # Get test IDs if available
        try:
            test_data = self.loader.get_test_data()
            if 'ID' in test_data.columns:
                submission['ID'] = test_data['ID']
            else:
                submission['ID'] = range(len(submission))
        except (AttributeError, KeyError):
            print("Warning: No test data IDs found, using sequential IDs")
            submission['ID'] = range(len(submission))
        
        print(f"Saving Predictions to {path}")
        submission.to_csv(path, index=False)
        print("Predictions Saved")


if __name__ == "__main__":
    try:
        pipeline = MultiTaskPipeline()
        pipeline.get_submission(os.path.join('submissions', 'multi_task_nn.csv'))
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()