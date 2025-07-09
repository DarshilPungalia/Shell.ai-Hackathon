import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


class DataFrameLoader:
    def __init__(self):
        self.trainData = pd.read_csv(r'dataset\train.csv')
        self.testData = pd.read_csv(r'dataset\test.csv')
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        
    @staticmethod
    def unskew(labels):
        labels_in = labels.copy()
        unskewer = PowerTransformer(method='yeo-johnson')
        unskewed = unskewer.fit_transform(labels_in)

        return unskewed, unskewer
                
    @staticmethod
    def generate_synthetic_data(dataframe, no_of_samples=3000):
        print('Generating Synthetic Data...')
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(dataframe)
        synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
        synthesizer.fit(dataframe)

        return synthesizer.sample(no_of_samples)

    def load(self, split_labels:bool = False, add_synthetic:bool = False):
        print('Loading data...')
        combinedData = self.trainData
        if add_synthetic:
            synData  = self.generate_synthetic_data(self.trainData)
            combinedData = pd.concat([self.trainData, synData], axis=0, ignore_index=True)

        self.X = combinedData.drop(columns=self.labels).to_numpy()
        self.Y = combinedData[self.labels]

        self.Y, skew_transformer = self.unskew(self.Y)
        self.skew_transformer = skew_transformer

        print(f'Shape of feature matrix is {self.X.shape}')
        print(f'Shape of label matrix is {self.Y.shape}')

        test = self.testData.drop(columns=['ID']).to_numpy()
        print(f'Shape of Test Data is {test.shape}')

        if split_labels:
            labelList = []
            for i in range(self.Y.shape[1]):
                labelList.append(self.Y[:, i])
            return self.X, labelList, test
        
        return self.X, self.Y, test

    
    def get_transformers(self):
        return self.skew_transformer
    
    def get_test_data(self):
        return self.testData