import pandas as pd
import numpy as np
from tab_pipeline import TabPFNPipeline
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from tqdm import tqdm
import os

class BayesianRidgePipeline:
    def __init__(self):
        self.models = []
        self.labels = [f'BlendProperty{i}' for i in range(1,11)]
        self.ridge = BayesianRidge
        print('Getting features from TabPFN...')
        self.features = TabPFNPipeline()
        self.x, self.y_split, self.x_test = self.features.get_full_features()

    
    def train(self):
        pbar = tqdm(zip(self.x, self.y_split, self.labels), total=len(self.labels))

        for x, y, label in pbar:
            pbar.set_description(f'Training Ridge for {label}')

            model = self.ridge()
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=24)
            model.fit(x_train, y_train)

            val_preds = model.predict(x_val)
            pbar.write(f'MAPE for {label} is {mape(y_val, val_preds)}')

            self.models.append(model)
    
        print(f'Finished Training {len(self.models)} Bayesian Ridge(s)')
    
    def predict(self)->np.ndarray:
        preds = []

        pbar = tqdm(zip(self.models, self.x_test, self.labels), total=len(self.models))
        for model, x_test, label in pbar:
            pbar.set_description(f'Predicting for {label}')

            pred = model.predict(x_test)
            preds.append(pred)
        

        return np.column_stack(preds)
    
    def get_submission(self, filename:str):
        sub = pd.DataFrame(data=range(1,501), columns=['ID'])
        sub[self.labels] = self.predict()

        path = os.path.join('submissions', filename)
        print(f'Saving Predictions to {path}')
        sub.to_csv(path, index=False)


if __name__ == "__main__":
    model = BayesianRidgePipeline()
    model.train()
    model.get_submission('bayesian.csv')