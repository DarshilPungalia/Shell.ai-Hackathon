import pandas as pd
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
import numpy as np
from ingestion import DataFrameLoader
from huggingface_hub import hf_hub_download
from typing import Literal
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings(action='ignore')

class TabPFNPipeline:
    def __init__(self, output_type: Literal['mean', 'median', 'mode'] = None):
        self.output_type = output_type or 'mean'
        self.models = []
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        self.loader = DataFrameLoader()
        print('Downloading Model from HuggingFace...')
        self.path = hf_hub_download(repo_id='Prior-Labs/TabPFN-v2-reg',
                               filename='tabpfn-v2-regressor.ckpt',
                               cache_dir=r'models')
        print(f'Model saved at {self.path}')
        self.regressor = TabPFNRegressor
        self.train()


    def train(self):
        self.x, self.y_split, self.x_test = self.loader.load(split_labels=True)
        pbar = tqdm(zip(self.y_split, self.labels), total=len(self.y_split), desc=f'Training {self.regressor.__name__}')
        for y, label in pbar:
            pbar.set_description(f'Training {self.regressor.__name__} for {label}')
            print(f'Shape of target: {y.shape}')

            x_train, x_val, y_train, y_val = train_test_split(self.x, y, test_size=0.2, random_state=69)

            print(f'Shape of x_train: {x_train.shape}')
            print(f'Shape of x_val: {x_val.shape}')
            print(f'Shape of y_train: {y_train.shape}')
            print(f'Shape of y_val: {y_val.shape}')

            model = self.regressor(device='auto', model_path=r'models\models--Prior-Labs--TabPFN-v2-reg\snapshots\213f8e38ec399a2a385fa46cab6f22b95cd90de8\tabpfn-v2-regressor.ckpt')
            model.fit(x_train, y_train)

            val_preds = model.predict(x_val, output_type=self.output_type)
            print(f'MAPE for {label}: {mape(y_val, val_preds)}')
            print(f'R2 Score for {label}: {r2(y_val, val_preds)}')

            self.models.append(model)
        
        print(f'Finished Traning {len(self.models)} {self.regressor.__name__}(s)')

    def get_submission(self, path):
        preds = []
        for model in tqdm(self.models, desc=f'Generating Predictions for each Blend using {self.output_type}', total=len(self.models)):
            pred = model.predict(self.x_test, output_type = self.output_type)
            preds.append(pred)
        
        preds = np.column_stack(preds)
        print(f'Stacked predictions shape: {preds.shape}')

        preds = self.loader.get_transformers().inverse_transform(preds)

        submission = pd.DataFrame(data=range(1, 501), columns=['ID'], index=None)
        submission[self.labels] = preds

        print(f'Saving Predictions to {path}')
        submission.to_csv(path, index=False)
        print('Predictions Saved')


if __name__ == "__main__":
    model = TabPFNPipeline(output_type='mean')
    model.get_submission(os.path.join('submissions', 'tabpfn_mean.csv'))
