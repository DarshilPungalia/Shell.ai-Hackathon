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
from sklearn.preprocessing import OneHotEncoder

class DeepGBM:
    def __init__(self, gradientBoosting_models: list | type = None):
        self.gradient_models = gradientBoosting_models or XGBRegressor
        self.model = None
        self.best_params = []
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        self.loader = DataFrameLoader()
        self.encoder = OneHotEncoder
        self.encoders = []


        #self.train()

    def oneHot(self, forest:list, test_set: bool = False):
        tree_encoded = []
        
        if not isinstance(forest, list):
            raise TypeError(f'forest needs to be of type list, got {type(forest)}')
        
        if not test_set:
            for tree in forest:
                encoder = self.encoder()
                encoded = encoder.fit_transform(tree)
                tree_encoded.append(encoded)
                self.encoders.append(encoder)
            print('OneHot encoded all the Trees')
        else:
            if len(self.encoders) != len(forest):
                raise ValueError(f'Number of encoders ({len(self.encoders)}) does not match number of trees ({len(forest)})')
            
            for tree, encoder in zip(forest, self.encoders):
                encoded = encoder.transform(tree)
                tree_encoded.append(encoded)
            print('OneHot encoded all the test Trees')
        
        return tree_encoded

    @staticmethod
    def combineBlendLeaves(forest:list):
        if not isinstance(forest, list):
            raise TypeError(f"Can't unpack {type(forest)} to combine for blends.")
        
        if len(forest) < 2:
            raise ValueError("Need at least 2 models to combine blends")
        
        combinedBlend = []
        
        for tree1, tree2 in zip(forest[0], forest[1]):
            if tree1.shape[0] != tree2.shape[0]:
                raise ValueError(f'Shape of all inputs across first dimension should be same. Got {tree1.shape[0]}, {tree2.shape[0]}')
            blendForest = np.column_stack([tree1, tree2])
            combinedBlend.append(blendForest)
        
        print(f'Combined {len(combinedBlend)} Blends from leaves of {len(forest)} models and shape of individual blend is {combinedBlend[0].shape}')
        
        return combinedBlend

    def gradientBoosting(self):
        forest, test_forest = [], []
        if isinstance(self.gradient_models, list):
            for booster in tqdm(self.gradient_models, desc='Extracting Leaves', total=len(self.gradient_models)):
                model = GradientBoostingPipeline(regressor=booster, n_trials=1)
                trees, test_trees = model.get_leaves()
                print(f'Shape of Trees is {trees[0].shape}')
                print(f"\n\n{trees[0]}\n\n")
                forest.append(trees)
                test_forest.append(test_trees)
            
            print(f'Extracted Leaves from {len(forest)} models.')
            forest, test_forest = self.combineBlendLeaves(forest), self.combineBlendLeaves(test_forest)
        
        elif self.gradient_models in (XGBRegressor, LGBMRegressor):
            model = GradientBoostingPipeline(regressor=self.gradient_models, n_trials=1)
            trees, test_trees = model.get_leaves()
            print(f'Shape of Trees is {trees[0].shape}.')
            forest.append(trees)
            test_forest.append(test_trees)
            
            print(f'Extracted Leaves from model')
        
        else:
            raise ValueError('Unidentified Gradient Boosting Model')
        
        forest = self.oneHot(forest)
        test_forest = self.oneHot(test_forest, test_set=True)
        
        return forest, test_forest
    

if __name__ == "__main__":
    model = DeepGBM(gradientBoosting_models= [XGBRegressor, LGBMRegressor])
    model.gradientBoosting()