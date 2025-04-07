
import pandas as pd     
import numpy as np    
import json  
import torch
from torch.utils.data import Dataset, DataLoader


class DataLoad(object):
    def __init__(self, glucose=True):
        self.glucose = glucose
        
    def load_data(self):
        if self.glucose:
            target = pd.read_excel("./dataset/target_substrate_waste_glucose.xlsx")
            source = pd.read_excel("./dataset/source_substrate_glucose_glucose.xlsx")
        else:
            target = pd.read_excel("./dataset/target_substrate_waste_lacticacid.xlsx")
            source = pd.read_excel("./dataset/source_substrate_glucose_lacticacid.xlsx")

        self.X_source, self.y_source, self.wl = DataLoad.get_X_y_from_df(source)
        self.X_target, self.y_target, _ = DataLoad.get_X_y_from_df(target)
        
        source_dataset = TorchData(self.X_source, self.y_source)
        target_dataset = TorchData(self.X_target, self.y_target)  # Target doesn't have labels

        # Step 4: Create DataLoader objects
        batch_size = 8

        self.source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
        self.target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
        
    def get_X_y_from_df(df):
        X = df.values[:,1:-1]
        y = df.values[:,-1]
        
        # avoid some features if the interval is smaller than 1 nm, e.g. from 1106.01 nm and 1106.97 nm, take one
        wl = np.array([int(float(i)) for i in df.columns[1:-1]])
        # Find unique values and their indices
        wl, unique_indices = np.unique(wl, return_index=True)
        X = X[:,unique_indices]
        return X, y, wl
    def load_class_data(self):
        # Opening JSON file
        f = open('./data/lacticacid_data.json')
        data = json.load(f)
        f.close()
        data['cs'].keys()
        self.x_train_c = np.array(data['classification']['ftir']['x_train'])
        self.x_test_c  = np.array(data['classification']['ftir']['x_test'])
        self.y_train_c = np.array(data['classification']['ftir']['y_train'])
        self.y_test_c  = np.array(data['classification']['ftir']['y_test'])
        self.wl_c      = np.array(data['classification']['ftir']['wl'])
    
    
    


# Step 1: Create custom datasets for source and target data
class TorchData(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.labels is not None:
            return self.data[index], self.labels[index]
        else:
            return self.data[index]
    
    
    


if __name__ == "__main__":
    D = DataLoad()
    D.load_data