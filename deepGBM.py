from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from ingestion import DataFrameLoader
from gradient_boosting_pipeline import GradientBoostingPipeline
from tqdm import tqdm

class GradientBoosting:
    def __init__(self, gradient_boosting_models: list | type = None):
        self.gradient_models = gradient_boosting_models or XGBRegressor
        self.model = None
        self.best_params = []
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        self.loader = DataFrameLoader()

    @staticmethod
    def combineBlendLeaves(forest: list):
        if not isinstance(forest, list) or len(forest) < 2:
            raise ValueError("Need at least 2 models to combine blends")
        
        combinedBlend = []
        
        for i in range(len(forest[0])):  
            combined_features = []
            for model_trees in forest: 
                combined_features.append(model_trees[i])
            
            blendForest = np.column_stack(combined_features)
            combinedBlend.append(blendForest)
        
        return combinedBlend

    def gradientBoosting(self):
        forest, test_forest = [], []
        if isinstance(self.gradient_models, list):
            for booster in tqdm(self.gradient_models, desc='Extracting Leaves', total=len(self.gradient_models)):
                model = GradientBoostingPipeline(regressor=booster, n_trials=30)
                trees, test_trees, y = model.get_leaves()
                print(f'Shape of Trees is {trees[0].shape}')
                forest.append(trees)
                test_forest.append(test_trees)
            
            print(f'Extracted Leaves from {len(forest)} models.')
            forest, test_forest = self.combineBlendLeaves(forest), self.combineBlendLeaves(test_forest)
        
        elif self.gradient_models in (XGBRegressor, LGBMRegressor):
            model = GradientBoostingPipeline(regressor=self.gradient_models, n_trials=30)
            trees, test_trees, y = model.get_leaves()
            print(f'Shape of Trees is {trees[0].shape}.')
            forest.append(trees)
            test_forest.append(test_trees)
            
            print(f'Extracted Leaves from model')
        
        else:
            raise ValueError('Unidentified Gradient Boosting Model')
                
        return forest, y, test_forest

class DeepNet:
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs
        self.model = None

    def build_model(self, leaf_inputs, num_outputs, embedding_dim=64, dropout_rate=0.2):
        self.num_models = len(leaf_inputs)
        max_leaf_index = int(max([leaf.max() for leaf in leaf_inputs])) + 1  

        model_inputs = []
        model_embeddings = []

        for i in range(self.num_models):
            num_trees_this_model = leaf_inputs[i].shape[1]
            print(f"Model {i}: {num_trees_this_model} trees")
            
            inp = tf.keras.layers.Input(shape=(num_trees_this_model,), dtype='int32', name=f'model_{i}_input')
            emb = tf.keras.layers.Embedding(input_dim=max_leaf_index, output_dim=embedding_dim, name=f'model_{i}_embed')(inp)
            pooled = tf.keras.layers.GlobalAveragePooling1D(name=f'model_{i}_pool')(emb)  
            model_inputs.append(inp)
            model_embeddings.append(pooled)


        # Concatenate embeddings from all models
        x = tf.keras.layers.Concatenate(name='concat_embeddings')(model_embeddings)        
        x = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        out = tf.keras.layers.Dense(num_outputs, name='output')(x)

        model = tf.keras.Model(inputs=model_inputs, outputs=out, name='EmbeddingMultiOutput')
        return model
    
    def train(self, leaf_inputs, y):
        leaf_inputs = [leaf.astype(np.int32) for leaf in leaf_inputs]
        model = self.build_model(leaf_inputs, num_outputs=self.num_outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='mse',
                      metrics=['mape'])
        
        model.summary()

        print('Training DNN...')
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True, min_delta=0.001)]
        model.fit(leaf_inputs,
                    y,
                    validation_split=0.25,
                    epochs=30,
                    batch_size=16,
                    callbacks=callbacks)
        
        self.model = model

    def predict(self, x:list):
        if not isinstance(x, list):
            raise TypeError(f'Input should be a list of lenght {self.num_models}, got type {type(x)}')

        if len(x) != self.num_models:
            raise ValueError(f'Expected input lenght is {self.num_models}, got {len(x)}')

        return self.model.predict(x)


class DeepGBM:
    def __init__(self, gradient_models: list | type = None, num_outputs: int = 10):
        self.gradient_models = gradient_models
        self.num_outputs = num_outputs
        self.model = None
    
    def train(self):
        print('Getting leaves from Gradient Boosting Models...')
        gradient_booster = GradientBoosting(gradient_boosting_models=self.gradient_models)
        x, y, self.x_test = gradient_booster.gradientBoosting()

        y = np.column_stack(y)

        print('Training the Neural Net on leaves...')
        self.model = DeepNet(num_outputs=10)
        self.model.train(x, y)
    
    def predict(self):
        return self.model.predict(self.x_test)

if __name__ == "__main__":
    models = [XGBRegressor, LGBMRegressor]
    deep_gbm = DeepGBM(gradient_models=models, num_outputs=10)
    history = deep_gbm.train()
    predictions = deep_gbm.predict()
    
    submission = pd.DataFrame(data=range(1, 501), columns=['ID'])

    submission[[f'BlendProperty{i+1}' for i in range(10)]] = predictions
    submission.to_csv(os.path.join('submissions', 'deepgbm.csv'), index=False)