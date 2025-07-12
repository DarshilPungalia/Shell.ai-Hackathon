from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
from ingestion import DataFrameLoader
from gradient_boosting_pipeline import GradientBoostingPipeline
from tqdm import tqdm

class DeepGBM:
    def __init__(self, gradient_boosting_models: list | XGBRegressor | LGBMRegressor = None):
        self.gradient_models = gradient_boosting_models or XGBRegressor
        self.model = None
        self.best_params = []
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        self.loader = DataFrameLoader()


        #self.train()

    def gradientBoosting(self):
        forest = []
        if isinstance(self.gradient_models, list):
            for booster in tqdm(self.gradient_models, desc='Extracting Leaves', total=len(self.gradient_models)):
                model = GradientBoostingPipeline(regressor=booster, n_trials=1)
                trees = model.get_leaves()
                print(f'Shape of Trees is {trees[0].shape}')
                forest.append(trees)
            
            print(f'Extracted Leaves from {len(forest)} models')
        
        if self.gradient_models in (XGBRegressor, LGBMRegressor):
            model = GradientBoostingPipeline(regressor=self.gradient_models, n_trials=1)
            trees = model.get_leaves()
            print(f'Shape of Trees is {trees[0].shape}')
            forest.append(trees)
            
            print(f'Extracted Leaves from model')
        
        else:
            raise ValueError('Unidentified Gradient Boosting Model')
        
        return forest



if __name__ == "__main__":
    model = DeepGBM(gradient_boosting_models= XGBRegressor)
    model.gradientBoosting()