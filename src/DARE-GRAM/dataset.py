from torch.utils.data import SubsetRandomSampler, Dataset, random_split, DataLoader
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SugarDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, source_domain, test_ratio=0.15):
        self.all_data = pd.read_excel(data_path)
        self.source_domain = 0 if source_domain else 1
        self.train_dataset, self.test_dataset = self._split_dataset(self.all_data, test_ratio)

    # Splitting the datasets into training and test sets
    def _split_dataset(self, dataset, test_ratio):
        dataset_size = len(dataset)
        test_size = int(test_ratio * dataset_size)
        train_size = dataset_size - test_size
        train_indices, test_indices = random_split(range(dataset_size), [train_size, test_size])

        return SugarSubset(dataset.iloc[list(train_indices)], self.source_domain), SugarSubset(dataset.iloc[list(test_indices)], self.source_domain)


class SugarSubset(Dataset):
    def __init__(self, data, source_domain):
        self.data = data
        self.source_domain = source_domain

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):        
        x = self.data.iloc[index, 1:-1].values
        y_reg = self.data.iloc[index, -1]
        y_domain = self.source_domain 

        x = torch.tensor(x, dtype=torch.float32)
        y_reg = torch.tensor(y_reg, dtype=torch.float32)
        y_domain = torch.tensor(y_domain, dtype=torch.float32)

        return x, y_reg, y_domain


class CombinedSugarDataset(Dataset):
    def __init__(self, source_data, target_data, source_domain, target_domain):
        if target_data.empty:
            self.data = source_data.copy()
        else:
            self.data = pd.DataFrame(np.vstack((source_data.values, target_data.values)), columns=source_data.columns)

        # Create corresponding domain labels (0 for source, 1 for target)
        source_domains = [source_domain] * len(source_data)
        target_domains = [target_domain] * len(target_data)
        self.domain_labels = source_domains + target_domains

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data.iloc[index, 1:-1].values  # Feature data
        y_reg = self.data.iloc[index, -1]       # Regression label
        y_domain = self.domain_labels[index]    # Domain label

        x = torch.tensor(x, dtype=torch.float32)
        y_reg = torch.tensor(y_reg, dtype=torch.float32)
        y_domain = torch.tensor(y_domain, dtype=torch.float32)

        return x, y_reg, y_domain
    



# class SugarDataset(torch.utils.data.Dataset):
#     def __init__(self, data_path, source_domain):
#         self.data = pd.read_excel(data_path)
#         self.source_domain = 0 if source_domain else 1

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         x = self.data.iloc[index, 1:-1].values
#         y_reg = self.data.iloc[index, -1]
#         y_domain = self.source_domain 

#         x = torch.tensor(x, dtype=torch.float32)
#         y_reg = torch.tensor(y_reg, dtype=torch.float32)
#         y_domain = torch.tensor(y_domain, dtype=torch.float32)

#         return x, y_reg, y_domain


# # Splitting the datasets into training and test sets
# def split_dataset(dataset, test_ratio, seed=2024):

#     torch.manual_seed(seed)

#     dataset_size = len(dataset)
#     test_ratio = int(test_ratio * dataset_size)
#     train_size = dataset_size - test_ratio
#     return random_split(dataset, [train_size, test_ratio])


# # # For glusoce, source domain, target domain
# source_glucose_dataset = SugarDataset(data_path='/mnt/beegfs/home/liu15/la-ood-tl-main/data/ss_glucose.xlsx', 
#                                             source_domain=True, 
#                                         )
# target_glucose_dataset = SugarDataset(data_path='/mnt/beegfs/home/liu15/la-ood-tl-main/data/cs_glucose.xlsx', 
#                                             source_domain=False, 
#                                         )

# # Splitting trainning set and test set
# source_train_dataset, source_test_dataset = split_dataset(source_glucose_dataset, test_ratio=0.2)
# target_train_dataset, target_test_dataset = split_dataset(target_glucose_dataset, test_ratio=0.2)

# target_whole_dataset, _ = split_dataset(target_glucose_dataset, test_ratio=0)  # For inference, evluating on the whole target set


# batch_size = 16
# num_workers = 8

# source_glucose_train_loader = DataLoader(
#     source_train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers
# )

# source_glucose_test_loader = DataLoader(
#     source_test_dataset,
#     shuffle=False,
#     batch_size=batch_size,
#     num_workers=num_workers
# )


# target_glucose_train_loader = DataLoader(
#     target_train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers
# )

# target_glucose_test_loader = DataLoader(
#     target_test_dataset,
#     shuffle=False,
#     batch_size=batch_size,
#     num_workers=num_workers
# )

# target_glucose_whole_loader = DataLoader(
#     target_whole_dataset,
#     shuffle=False,
#     batch_size=batch_size,
#     num_workers=num_workers
# )


# # # For lacticacid, source domain, target domain
# source_lacticacid_dataset = SugarDataset(data_path='/mnt/beegfs/home/liu15/la-ood-tl-main/data/ss_lacticacid.xlsx', 
#                                             source_domain=True, 
#                                         )
# target_lacticacid_dataset = SugarDataset(data_path='/mnt/beegfs/home/liu15/la-ood-tl-main/data/cs_lacticacid.xlsx', 
#                                             source_domain=False, 
#                                         )

# # Splitting trainning set and test set
# source_train_dataset, source_test_dataset = split_dataset(source_lacticacid_dataset, test_ratio=0.2)
# target_train_dataset, target_test_dataset = split_dataset(target_lacticacid_dataset, test_ratio=0.2)

# target_whole_dataset, _ = split_dataset(target_lacticacid_dataset, test_ratio=0)  # For inference, evluating on the whole target set


# batch_size = 16
# num_workers = 8

# source_lacticacid_train_loader = DataLoader(
#     source_train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers
# )

# source_lacticacid_test_loader = DataLoader(
#     source_test_dataset,
#     shuffle=False,
#     batch_size=batch_size,
#     num_workers=num_workers
# )


# target_lacticacid_train_loader = DataLoader(
#     target_train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers
# )

# target_lacticacid_test_loader = DataLoader(
#     target_test_dataset,
#     shuffle=False,
#     batch_size=batch_size,
#     num_workers=num_workers
# )

# target_lacticacid_whole_loader = DataLoader(
#     target_whole_dataset,
#     shuffle=False,
#     batch_size=batch_size,
#     num_workers=num_workers
# )
