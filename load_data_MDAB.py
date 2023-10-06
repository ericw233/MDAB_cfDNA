import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

from pad_and_reshape import pad_and_reshape, pad_and_reshape_1D

def load_data(data_dir="/mnt/binf/eric/Mercury_June2023_new/Feature_all_June2023_R01BMatch_ClusterKAG9_SeqBatch.csv", input_size=950, feature_type = "Arm"):
    # Read data from CSV file
    data = pd.read_csv(data_dir).dropna(axis=1)

    # keep a full dataset without shuffling
    mapping = {'Healthy':0,'Cancer':1}
    X_all = data.filter(regex = feature_type, axis=1).dropna(axis=1)
    y_all = data.loc[:,'Train_Group'].replace(mapping)
    d_all = data.loc[:,'Domain']
    b_all = data.loc[:,'Batch']
    
    # Split the data into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train = data.loc[data["train"] == "training"].filter(regex = feature_type, axis=1)
    y_train = data.loc[data["train"] == "training","Train_Group"].replace(mapping)
    d_train = data.loc[data["train"] == "training","Domain"]
    b_train = data.loc[data["train"] == "training","Batch"]
    
    X_test = data.loc[data["train"] == "validation"].filter(regex = feature_type, axis=1)
    y_test = data.loc[data["train"] == "validation","Train_Group"].replace(mapping)
    d_test = data.loc[data["train"] == "validation","Domain"]
    b_test = data.loc[data["train"] == "validation","Batch"]
    
    # Scale the features to a suitable range (e.g., [0, 1])
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X_all)

    # Convert the data to PyTorch tensors
    input_size = input_size
    X_train_tensor = pad_and_reshape(X_train_scaled, input_size).type(torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    d_train_tensor = torch.tensor(d_train.values, dtype=torch.float32)
    b_train_tensor = torch.tensor(b_train.values, dtype=torch.int)
    
    X_test_tensor = pad_and_reshape(X_test_scaled, input_size).type(torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    d_test_tensor = torch.tensor(d_test.values, dtype=torch.float32)
    b_test_tensor = torch.tensor(b_test.values, dtype=torch.int)
    
    X_all_tensor = pad_and_reshape(X_all_scaled, input_size).type(torch.float32)
    y_all_tensor = torch.tensor(y_all.values, dtype=torch.float32)
    d_all_tensor = torch.tensor(d_all.values, dtype=torch.float32)
    b_all_tensor = torch.tensor(b_all.values, dtype=torch.int)
    
    ### keep unshuffled X_train
    # X_train_tensor_unshuffled = pad_and_reshape(X_train_scaled, input_size).type(torch.float32)
    # y_train_tensor_unshuffled = torch.tensor(y_train.values, dtype=torch.float32)
    train_sampleid = data.loc[data["train"] == "training","SampleID"].values

    return data, X_train_tensor, y_train_tensor, d_train_tensor, b_train_tensor, X_test_tensor, y_test_tensor, d_test_tensor, b_test_tensor, X_all_tensor, y_all_tensor, d_all_tensor, b_all_tensor, train_sampleid


def load_data_1D(data_dir="/mnt/binf/eric/Mercury_June2023_new/Feature_all_June2023_R01BMatch.csv", input_size=900, feature_type = "Arm"):
    # Read data from CSV file
    data = pd.read_csv(data_dir).dropna(axis=1)

    # keep a full dataset without shuffling
    mapping = {'Healthy':0,'Cancer':1}
    X_all = data.filter(regex = feature_type, axis=1).dropna(axis=1)
    y_all = data.loc[:,'Train_Group'].replace(mapping)
    d_all = data.loc[:,'Domain']
    b_all = data.loc[:,'Batch']
    
    # Split the data into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train = data.loc[data["train"] == "training"].filter(regex = feature_type, axis=1)
    y_train = data.loc[data["train"] == "training","Train_Group"].replace(mapping)
    d_train = data.loc[data["train"] == "training","Domain"]
    b_train = data.loc[data["train"] == "training","Batch"]
    
    X_test = data.loc[data["train"] == "validation"].filter(regex = feature_type, axis=1)
    y_test = data.loc[data["train"] == "validation","Train_Group"].replace(mapping)
    d_test = data.loc[data["train"] == "validation","Domain"]
    b_test = data.loc[data["train"] == "validation","Batch"]
    
    # Scale the features to a suitable range (e.g., [0, 1])
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X_all)

    # Convert the data to PyTorch tensors
    input_size = input_size
    X_train_tensor = pad_and_reshape_1D(X_train_scaled, input_size).type(torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    d_train_tensor = torch.tensor(d_train.values, dtype=torch.float32)
    b_train_tensor = torch.tensor(b_train.values, dtype=torch.long)
    
    X_test_tensor = pad_and_reshape_1D(X_test_scaled, input_size).type(torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    d_test_tensor = torch.tensor(d_test.values, dtype=torch.float32)
    b_test_tensor = torch.tensor(b_test.values, dtype=torch.long)
    
    X_all_tensor = pad_and_reshape_1D(X_all_scaled, input_size).type(torch.float32)
    y_all_tensor = torch.tensor(y_all.values, dtype=torch.float32)
    d_all_tensor = torch.tensor(d_all.values, dtype=torch.float32)
    b_all_tensor = torch.tensor(b_all.values, dtype=torch.long)
    
    ### keep unshuffled X_train
    # X_train_tensor_unshuffled = pad_and_reshape(X_train_scaled, input_size).type(torch.float32)
    # y_train_tensor_unshuffled = torch.tensor(y_train.values, dtype=torch.float32)
    train_sampleid = data.loc[data["train"] == "training","SampleID"].values

    return data, X_train_tensor, y_train_tensor, d_train_tensor, b_train_tensor, X_test_tensor, y_test_tensor, d_test_tensor, b_test_tensor, X_all_tensor, y_all_tensor, d_all_tensor, b_all_tensor, train_sampleid


def load_data_1D_impute(data_dir="/mnt/binf/eric/Mercury_Aug2023_new/Feature_all_Aug2023_DomainKAG9v1.csv", input_size=900, feature_type = "Arm"):
    # Read data from CSV file
    data = pd.read_csv(data_dir)
    data['combine_group'] = data['Train_Group'].astype(str)+'-'+data['Domain'].astype(str)+'-'+data['Batch'].astype(str)
    
    # keep a full dataset without shuffling
    mapping = {'Healthy':0,'Cancer':1}
    
    # Split the data into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train = data.loc[data["train"] == "training"].filter(regex = feature_type, axis=1)
    y_train = data.loc[data["train"] == "training","Train_Group"].replace(mapping)
    d_train = data.loc[data["train"] == "training","Domain"]
    b_train = data.loc[data["train"] == "training","Batch"]
    
    train_split = StratifiedShuffleSplit(n_splits=1,test_size=0.25,random_state=42)
    for train_index, valid_index in train_split.split(X_train, X_train['combine_group']):
        X_train_train = X_train.iloc[train_index]
        y_train_train = y_train.iloc[train_index]
        d_train_train = d_train.iloc[train_index]
        b_train_train = b_train.iloc[train_index]
        X_train_valid = X_train.iloc[valid_index]
        y_train_valid = y_train.iloc[valid_index]
        d_train_valid = d_train.iloc[valid_index]
        b_train_valid = b_train.iloc[valid_index]
        
    X_test = data.loc[data["train"] == "validation"].filter(regex = feature_type, axis=1)
    y_test = data.loc[data["train"] == "validation","Train_Group"].replace(mapping)
    d_test = data.loc[data["train"] == "validation","Domain"]
    b_test = data.loc[data["train"] == "validation","Batch"]
    
    X_all = data.filter(regex = feature_type, axis=1)
    y_all = data.loc[:,'Train_Group'].replace(mapping)
    d_all = data.loc[:,'Domain']
    b_all = data.loc[:,'Batch']
    
    #### drop constant NA columns based on X_train
    na_columns = X_train.columns[X_train.isna().all()]
    X_train_drop = X_train.drop(columns = na_columns)
    X_test_drop = X_test.drop(columns = na_columns)
    X_all_drop = X_all.drop(columns = na_columns)
    
    #### impute variables based on X_train_drop
    mean_imputer = SimpleImputer(strategy = 'mean')
    X_train_drop_imputed = mean_imputer.fit_transform(X_train_drop)
    X_test_drop_imputed = mean_imputer.transform(X_test_drop)
    X_all_drop_imputed = mean_imputer.transform(X_all_drop)
    
    # Scale the features to a suitable range (e.g., [0, 1])
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_drop_imputed)
    X_test_scaled = scaler.transform(X_test_drop_imputed)
    X_all_scaled = scaler.transform(X_all_drop_imputed)

    # Convert the data to PyTorch tensors
    input_size = input_size
    X_train_tensor = pad_and_reshape_1D(X_train_scaled, input_size).type(torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    d_train_tensor = torch.tensor(d_train.values, dtype=torch.float32)
    b_train_tensor = torch.tensor(b_train.values, dtype=torch.float32)
    
    X_test_tensor = pad_and_reshape_1D(X_test_scaled, input_size).type(torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    d_test_tensor = torch.tensor(d_test.values, dtype=torch.float32)
    b_test_tensor = torch.tensor(b_test.values, dtype=torch.float32)
    
    X_all_tensor = pad_and_reshape_1D(X_all_scaled, input_size).type(torch.float32)
    y_all_tensor = torch.tensor(y_all.values, dtype=torch.float32)
    d_all_tensor = torch.tensor(d_all.values, dtype=torch.float32)
    b_all_tensor = torch.tensor(b_all.values, dtype=torch.float32)
    
    ### keep unshuffled X_train
    # X_train_tensor_unshuffled = pad_and_reshape_1D(X_train_scaled, input_size).type(torch.float32)
    # y_train_tensor_unshuffled = torch.tensor(y_train.values, dtype=torch.float32)
    train_sampleid = data.loc[data["train"] == "training","SampleID"].values

    return data, X_train_tensor, y_train_tensor, d_train_tensor, b_train_tensor, X_test_tensor, y_test_tensor, d_test_tensor, b_test_tensor, X_all_tensor, y_all_tensor, d_all_tensor, b_all_tensor, train_sampleid



