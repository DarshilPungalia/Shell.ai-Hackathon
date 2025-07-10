import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
from ingestion import DataFrameLoader
import warnings

warnings.filterwarnings(action='ignore')

class PLSPipeline:
    def __init__(self):
        self.model = None
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        self.loader = DataFrameLoader()
        self.regressor = PLSRegression
        self.train()


    def train(self):
        X, Y, self.x_test = self.loader.load()
        print(f'Training {self.regressor.__name__}...')

        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=69)

        print(f'Shape of x_train: {x_train.shape}')
        print(f'Shape of x_val: {x_val.shape}')
        print(f'Shape of y_train: {y_train.shape}')
        print(f'Shape of y_val: {y_val.shape}')

        model = self.regressor(n_components=10)
        model.fit(x_train, y_train)

        val_preds = model.predict(x_val)
        print(f'MAPE: {mape(y_val, val_preds)}')
        print(f'R2 Score: {r2(y_val, val_preds)}')

        self.model = model
        print(f'Finished Traning {self.regressor.__name__}')

    def get_submission(self, path):
        preds = self.model.predict(self.x_test)
        
        preds = self.loader.get_transformers().inverse_transform(preds)

        submission = pd.DataFrame(data=range(1, 501), columns=['ID'], index=None)
        submission[self.labels] = preds

        print(f'Saving Predictions to {path}')
        submission.to_csv(path, index=False)
        print('Predictions Saved')

if __name__ == '__main__':
    model = PLSPipeline()
    model.get_submission(r'submissions/pls.csv')