import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.decomposition import KernelPCA
import numpy as np
from sklearn.preprocessing import RobustScaler

class DataFrameLoader:
    def __init__(self):
        self.trainData = pd.read_csv(r'dataset\train.csv')
        self.testData = pd.read_csv(r'dataset\test.csv')
        self.labels = ['BlendProperty1', 'BlendProperty2', 'BlendProperty3', 'BlendProperty4', 'BlendProperty5', 
                       'BlendProperty6', 'BlendProperty7', 'BlendProperty8', 'BlendProperty9','BlendProperty10']
        self.pca = KernelPCA(n_components=10, kernel='rbf')
        self.scaler = RobustScaler()

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
    
    def reduce_dims(self, df, reduce_test:bool = False):
        print('Reducing Feature Dimnesions')
        df_in = df.copy()
        reduce_columns = [f'Component{i}_Property{j}' for i in range(1, 6) for j in range(1, 11)]
        x = df_in[reduce_columns]
        df_in.drop(columns=reduce_columns, inplace=True)

        if reduce_test:
            x_transformed = self.pca.transform(x)
            df_in[[f'Component_Property{i}' for i in range(10)]] = x_transformed

            return df_in
        
        x_transformed = self.pca.fit_transform(x)

        df_in[[f'Component_Property{i}' for i in range(10)]] = x_transformed

        return df_in
    
    def add_features(self, df, test_set:bool = False):  
        print('Engineering new Features...')  
        df_engineered = df.copy()
        
        new_features = {}
        
        fraction_cols = [f'Component{i}_fraction' for i in range(1, 6)]
        
        property_cols = {}
        for prop_num in range(1, 11):
            property_cols[f'Property{prop_num}'] = [f'Component{i}_Property{prop_num}' for i in range(1, 6)]
        
        # 1. FRACTION-BASED FEATURES        
        new_features['fraction_diversity'] = df_engineered[fraction_cols].apply(
            lambda x: -np.sum(x * np.log(x + 1e-10)), axis=1)  # Shannon entropy
        
        dominant_indices = df_engineered[fraction_cols].idxmax(axis=1)
        new_features['dominant_component'] = dominant_indices.str.extract(r'Component(\d+)').astype(int).iloc[:, 0]
        
        new_features['max_fraction'] = df_engineered[fraction_cols].max(axis=1)
        new_features['min_fraction'] = df_engineered[fraction_cols].min(axis=1)
        new_features['fraction_std'] = df_engineered[fraction_cols].std(axis=1)
        new_features['fraction_range'] = new_features['max_fraction'] - new_features['min_fraction']
        
        new_features['effective_components'] = 1 / (df_engineered[fraction_cols] ** 2).sum(axis=1)
        
        for i, col in enumerate(fraction_cols, 1):
            new_features[f'has_component_{i}'] = (df_engineered[col] > 0.01).astype(int)
        
        new_features['num_active_components'] = sum(new_features[f'has_component_{i}'] for i in range(1, 6))
        
        # 2. WEIGHTED PROPERTY FEATURES (Volume-weighted averages)        
        for prop_name, prop_cols in property_cols.items():
            weighted_sum = sum(df_engineered[fraction_cols[i]] * df_engineered[prop_cols[i]] 
                            for i in range(5))
            new_features[f'weighted_avg_{prop_name}'] = weighted_sum
            
            weighted_mean = weighted_sum
            weighted_var = sum(df_engineered[fraction_cols[i]] * 
                            (df_engineered[prop_cols[i]] - weighted_mean) ** 2 
                            for i in range(5))
            new_features[f'weighted_std_{prop_name}'] = np.sqrt(weighted_var)
            
            new_features[f'range_{prop_name}'] = (df_engineered[prop_cols].max(axis=1) - 
                                                df_engineered[prop_cols].min(axis=1))
            
            new_features[f'cv_{prop_name}'] = (df_engineered[prop_cols].std(axis=1) / 
                                            (df_engineered[prop_cols].mean(axis=1) + 1e-10))
        
        # 3. COMPONENT-SPECIFIC FEATURES        
        for comp_num in range(1, 6):
            comp_prop_cols = [f'Component{comp_num}_Property{prop}' for prop in range(1, 11)]
            
            new_features[f'Component{comp_num}_prop_mean'] = df_engineered[comp_prop_cols].mean(axis=1)
            new_features[f'Component{comp_num}_prop_std'] = df_engineered[comp_prop_cols].std(axis=1)
            new_features[f'Component{comp_num}_prop_sum'] = df_engineered[comp_prop_cols].sum(axis=1)
            
            new_features[f'Component{comp_num}_contribution'] = (
                df_engineered[f'Component{comp_num}_fraction'] * 
                new_features[f'Component{comp_num}_prop_mean']
            )
        
        '''# 4. SYNERGY AND ANTAGONISM FEATURES        
        for prop_num in range(1, 11):
            prop_cols = [f'Component{i}_Property{prop_num}' for i in range(1, 6)]
            total_prop_contribution = sum(df_engineered[fraction_cols[i]] * 
                                        df_engineered[prop_cols[i]] for i in range(5))
            
            for i in range(5):
                new_features[f'Component{i+1}_Property{prop_num}_contribution_pct'] = (
                    df_engineered[fraction_cols[i]] * df_engineered[prop_cols[i]] / 
                    (total_prop_contribution + 1e-10)
                )'''
        
        # 5. BLEND COMPLEXITY FEATURES
        new_features['blend_complexity_score'] = (
            new_features['fraction_diversity'] * 
            new_features['num_active_components'] * 
            np.mean([new_features[f'weighted_std_Property{i}'] for i in range(1, 11)], axis=0)
        )
        
        property_similarities = []
        for prop_num in range(1, 11):
            prop_cols = [f'Component{i}_Property{prop_num}' for i in range(1, 6)]
            prop_std = df_engineered[prop_cols].std(axis=1)
            property_similarities.append(prop_std)
        
        new_features['blend_homogeneity'] = 1 / (1 + np.mean(property_similarities, axis=0))

        # COMBINE ALL FEATURES
        new_features_df = pd.DataFrame(new_features, index=df_engineered.index)
        df_engineered = pd.concat([df_engineered, new_features_df], axis=1)

        df_cols = df_engineered.columns
        if test_set:
            scaled = self.scaler.transform(df_engineered)
        else:
            scaled = self.scaler.fit_transform(df_engineered)
        
        df_scaled = pd.DataFrame(data=scaled, columns=df_cols)
        
        return df_scaled

    def load(self, split_labels:bool = False, add_synthetic:bool = False, reduce_dims:bool = False, add_features:bool = False):
        print('Loading data...')
        combinedData = self.trainData.drop(columns=self.labels)
        self.testData = self.testData.drop(columns=['ID'])
        if add_synthetic:
            synData  = self.generate_synthetic_data(self.trainData)
            combinedData = pd.concat([self.trainData, synData], axis=0, ignore_index=True)

        if add_features:
            combinedData = self.add_features(combinedData)
            self.testData = self.add_features(self.testData, test_set=True)
        
        if reduce_dims:
            combinedData = self.reduce_dims(combinedData)
            self.testData = self.reduce_dims(self.testData, reduce_test=True)
        
        self.X = combinedData.to_numpy()
        self.Y = self.trainData[self.labels]

        self.Y, skew_transformer = self.unskew(self.Y)
        self.skew_transformer = skew_transformer

        print(f'Shape of feature matrix is {self.X.shape}')
        print(f'Shape of label matrix is {self.Y.shape}')

        test = self.testData.to_numpy()
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