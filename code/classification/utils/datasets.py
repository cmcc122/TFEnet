import os
import warnings
from scipy.io import loadmat
import numpy as np
# import mne
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split,TensorDataset
import pandas as pd
import gc

import os
import sys


current_dir = os.path.abspath(os.path.dirname(__file__))

parent_dir = os.path.join(current_dir, "..")
classification_dir = os.path.join(parent_dir, "..") 
sys.path.append(classification_dir)

from classification.utils.parse_config import read_json_config
def mean_standardize_fit(X):
    m1 = np.mean(X, axis=1)
    mean = np.mean(m1, axis=0)

    s1 = np.std(X, axis=1)
    std = np.mean(s1, axis=0)

    return mean, std


def mean_standardize_transform(X, mean, std):
    return (X - mean) / std

class EEG_Drug_health:
    def __init__(self, root_directory, L=30720, D=30, n=1, L_parts=60):
        self.mat_files_directory0 = os.path.join(root_directory, 'health')
        self.mat_files_directory1 = os.path.join(root_directory, 'drug')
        self.L = L
        self.D = D
        self.n = n
        self.L_parts = L_parts
        self.L_part_size = L // L_parts

    def loadData(self, mat_files_directory):
        people_files = {}
        for file in list(sorted(os.listdir(mat_files_directory))):
            if file.endswith('.mat'):
                person_id = file[:-5]
                if person_id not in people_files:
                    people_files[person_id] = []

                    file_path1 = os.path.join(mat_files_directory, person_id + '1.mat')
                    data1 = loadmat(file_path1)
                    data1 = {k: v for k, v in data1.items() if not k.startswith('__')}
                    data1 = data1['epoch_data_reshaped']
                    people_files[person_id].append(data1)
                    # Clear memory for the first file
                    del data1
                    gc.collect()  # Trigger garbage collection

                    file_path2 = os.path.join(mat_files_directory, person_id + '2.mat')
                    data2 = loadmat(file_path2)
                    data2 = {k: v for k, v in data2.items() if not k.startswith('__')}
                    data2 = data2['epoch_data_reshaped']
                    people_files[person_id].append(data2)
                    # Clear memory for the second file
                    del data2
                    gc.collect()  # Trigger garbage collection
                    
        return people_files

    def dataConvert(self, data0, data1):
        data0 = list(data0.values())
        split_index_0 = int(len(data0) * 0.8)
        train0 = data0[:split_index_0]
        test0 = data0[split_index_0:]

        train0, test0 = np.concatenate(train0, axis=0), np.concatenate(test0, axis=0)
        train_label0, test_label0 = np.zeros(train0.shape[0]), np.zeros(test0.shape[0])

        data1 = list(data1.values())
        split_index_1 = int(len(data1) * 0.8)

        train1 = data1[:split_index_1]
        test1 = data1[split_index_1:]

        train1,  test1 = np.concatenate(train1, axis=0), np.concatenate(test1, axis=0)
        train_label1, test_label1 = np.ones(train1.shape[0]), np.ones(test1.shape[0])

   
        train_data = np.concatenate([train0, train1], axis=0)
        train_label = np.concatenate([train_label0, train_label1], axis=0)

        test_data = np.concatenate([test0, test1], axis=0)
        test_label = np.concatenate([test_label0, test_label1], axis=0)
        
        train_mean, train_std = mean_standardize_fit(train_data)
        train_data= mean_standardize_transform(train_data,train_mean, train_std)
        test_mean, test_std = mean_standardize_fit(test_data)
        test_data= mean_standardize_transform(test_data,test_mean, test_std)

        train_data = np.transpose(train_data, (0, 2, 1))
        test_data = np.transpose(test_data, (0, 2, 1))
        
        return train_data, train_label, test_data, test_label

    def slidWindows(self, x_tensor, y_tensor):
        x_tensor = torch.tensor(x_tensor, dtype=torch.float32)
        y_tensor = torch.tensor(y_tensor, dtype=torch.long)
        X_new = []
        y_new = []
        for i in range(len(y_tensor)):
            for j in range(self.L_parts):
                X_part = x_tensor[i, j * self.L_part_size:(j + 1) * self.L_part_size, :]

                X_new.append(X_part)
                y_new.append(y_tensor[i])

        X_new_tensor = torch.stack(X_new).unsqueeze(0).permute(1, 3, 0, 2)
        y_new_tensor = torch.tensor(y_new)

        return X_new_tensor, y_new_tensor

    def process_data(self):
        data0 = self.loadData(self.mat_files_directory0)
        data1 = self.loadData(self.mat_files_directory1)
        train_data, train_label, test_data, test_label = self.dataConvert(data0, data1)
        X_train, y_train = self.slidWindows(train_data, train_label)
        X_test, y_test = self.slidWindows(test_data, test_label)

        class_sample_counts = np.bincount(y_train.numpy())

        class_weights = 1. / class_sample_counts

        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        return train_dataset,  test_dataset,class_weights


class EEG_alchole_health:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.folder_a = os.path.join(root_folder, 'a')
        self.folder_c = os.path.join(root_folder, 'c')

    def load_data(self, csv_files_directory):
        people_data = {}

        for person_folder in list(sorted(os.listdir(csv_files_directory))):

            person_folder_path = os.path.join(csv_files_directory, person_folder)
            if os.path.isdir(person_folder_path):
                person_data = []
                for file in os.listdir(person_folder_path):
                    if file.endswith('.csv'):
                        file_path = os.path.join(person_folder_path, file)
                        df = pd.read_csv(file_path)  # Assuming CSV files have similar structure
                        person_data.append(df)
                if person_data:
                    people_data[person_folder] = person_data
        return people_data

    def data_convert(self, data0, data1):
        data0_items = list(data0.items())

        data0 = dict(data0_items)
        key0 = list(data0.keys())

        data1_items = list(data1.items())

        data1 = dict(data1_items)
        key1 = list(data1.keys())

        split_index_0 = int(len(key0) * 0.85)
        split_index_1 = int(len(key1) * 0.85)

        train_keys0 = key0[:split_index_0]
        test_keys0 = key0[split_index_0:]
        
        train_keys1 = key1[:split_index_1]
        test_keys1 = key1[split_index_1:]

        train0 = [data0[key] for key in train_keys0]
        test0 = [data0[key] for key in test_keys0]
        
        train1 = [data1[key] for key in train_keys1]
        test1 = [data1[key] for key in test_keys1]

        train0, test0 = np.concatenate(train0, axis=0), np.concatenate(test0, axis=0)
        train1, test1 = np.concatenate(train1, axis=0), np.concatenate(test1, axis=0)

        train_label0, test_label0 = np.zeros(train0.shape[0]), np.zeros(test0.shape[0])
        train_label1, test_label1 = np.ones(train1.shape[0]), np.ones(test1.shape[0])

        # Combine and reshape data
        train_data = np.concatenate([train0, train1], axis=0)
        train_label = np.concatenate([train_label0, train_label1], axis=0)

        test_data = np.concatenate([test0, test1], axis=0)#(batch_size,length,channel)
        test_label = np.concatenate([test_label0, test_label1], axis=0)

        train_mean, train_std = mean_standardize_fit(train_data)
        train_data= mean_standardize_transform(train_data,train_mean, train_std)
        test_mean, test_std = mean_standardize_fit(test_data)
        test_data= mean_standardize_transform(test_data,test_mean, test_std)

        train_data = np.transpose(train_data, (0, 2, 1))#(batch_size,channel,length)
        test_data = np.transpose(test_data, (0, 2, 1))
        
        train_data = np.expand_dims(train_data, axis=2)  # (batch_size, channel, 1, length)
        test_data = np.expand_dims(test_data, axis=2)    # (batch_size, channel, 1, length)

        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_label = torch.tensor(train_label, dtype=torch.long)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_label = torch.tensor(test_label, dtype=torch.long)
        

        class_sample_counts = np.bincount(train_label.numpy())
        print("Class sample counts:", class_sample_counts)

        class_weights = 1. / class_sample_counts
        class_weights = class_weights / class_weights.sum()  
        print("Class weights:", class_weights)

        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        print("Class weights tensor:", class_weights)

        return train_data, train_label, test_data, test_label, class_weights
    def process_data(self):    
        data0 = self.load_data(self.folder_c)
        print("数据0载入完毕")
        data1 = self.load_data(self.folder_a)
        print("数据1载入完毕")
        X_train, y_train, X_test, y_test,class_weights= self.data_convert(data0, data1)
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        return train_dataset,  test_dataset,class_weights

    

def load_and_process_dataset(root_directory,batch_size, val_batch_size):
    dataset = EEG_alchole_health(root_directory)
    train_dataset, test_dataset ,class_weights= dataset.process_data()
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader,class_weights




if __name__ == '__main__':
    # 参数读取
    custom_dict = read_json_config("E:\data\code\config\custom.json")
    assert custom_dict, "文件读取异常"
    training_environment_setting = custom_dict.get("training_environment_setting", {})
    training_process_setting = custom_dict.get("training_process_setting", {})
    hyperparameters = custom_dict.get("hyperparameters", {})
    pretraining_setting = custom_dict.get("pretraining", {})

    dataset_path = training_environment_setting.get("dataset", {})

    train_dataloader, test_dataloader,class_weights=load_and_process_dataset(dataset_path, batch_size=32, val_batch_size=32)
    print(class_weights)
    cont=0
    for batch in train_dataloader:
        cont+=1
        data, labels = batch
        print(data.shape)
    print(cont)
    cont=0
    
    cont=0
    for batch in test_dataloader:
        cont+=1
        data, labels = batch
        print(data.shape)
    print(cont)
