import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sdv.single_table import CTGANSynthesizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class DataLoader:
    def __init__(self):
        self.trainData = pd.read_csv(r'dataset\train.csv')
        self.testData = pd.read_csv(r'dataset\test.csv')
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        
        self.transformers = {}

    @staticmethod
    def unskew(labels):
        labels_in = labels.copy()
        unskewer = PowerTransformer(method='yeo-johnson')
        unskewed = unskewer.fit_transform(labels_in)

        return unskewed, unskewer
    
    @staticmethod
    def transform(data, columns, label:bool=False):
        scaler = MinMaxScaler()

        if label and isinstance(data, np.ndarray):
            transformed_data = scaler.fit_transform(data)
            return transformed_data, scaler
        
        basePipeline = Pipeline(steps=[
            ('scaler', scaler)
        ])

        transformer = ColumnTransformer(transformers=[
            ('scaler', basePipeline, columns)
        ])

        transformed_data = transformer.fit_transform(data)

        return transformed_data, transformer
            
    @staticmethod
    def generate_synthetic_data(dataframe, no_of_samples=1000):
        print('Generating Synthetic Data...')
        synthesizer = CTGANSynthesizer()
        synthesizer.fit(dataframe)

        return synthesizer.sample(no_of_samples)

    def load(self, split_labels:bool = False):
        print('Loading data...')
        synData  = self.generate_synthetic_data(self.trainData)

        combinedData = pd.concat([self.trainData, synData], axis=0, ignore_index=True)

        self.X = combinedData.drop(columns=self.labels)
        self.Y = combinedData[self.labels]

        self.Y, skew_transformer = self.unskew(self.Y)
        self.transformers['unskew_label'] = skew_transformer

        features, feature_transformer = self.transform(self.X, self.X.columns)
        labels, label_transformer = self.transform(self.Y, range(len(self.labels)), label=True)
        self.transformers['feature_transformer'] = feature_transformer
        self.transformers['label_transformer'] = label_transformer
        print(f'Shape of feature matrix is {features.shape}')
        print(f'Shape of label matrix is {labels.shape}')

        test = feature_transformer.transform(self.testData)
        print(f'Shape of Test Data is {test.shape}')

        if split_labels:
            labelList = []
            for i, label in enumerate(self.labels):
                labelList.append(labels[:, i])
            return features, labelList, test
        
        return features, labels, test

    
    def get_transformers(self):
        return self.transformers