import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import ADASYN, SMOTE

class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# def prepare_oversampling_dataset(X_train, y_train, X_test, y_test, train_ratio, val_ratio, batch_size, scaler=None):
#     # if scaler is None:
#     #     scaler = StandardScaler()
#     # X_scaled = scaler.fit_transform(X)
#     # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=True, test_size=0.2, stratify=y)
#     # 对训练集执行过采样操作
#     strategy = {0: 350, 1:350}
#     adasyn = ADASYN(random_state=42, sampling_strategy=strategy)
#     X_train, y_train = adasyn.fit_resample(X_train, y_train)
#     train_dataset = Dataset(X_train, y_train)
#     test_dataset = Dataset(X_test, y_test)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     return {
#         'datasets': {
#             'train': train_dataset,
#             'test': test_dataset
#         },
#         'loaders': {
#             'train': train_loader,
#             'test': test_loader
#         }
#     }
    
def prepare_datasets(X_train, y_train, X_test, y_test, train_ratio, val_ratio, batch_size, scaler=None, X=None, y=None, Mode='Train'):
    if Mode == 'Train':
        train_dataset = Dataset(X_train, y_train)
        test_dataset = Dataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        return {
            'datasets': {
                'train': train_dataset,
                # 'val': val_dataset,
                'test': test_dataset
            },
            'loaders': {
                'train': train_loader,
                # 'val': val_loader,
                'test': test_loader
            }
        }
    elif Mode == 'Explain':
        dataset = Dataset(X, y)
        dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        return {
            'loaders': {
                'data': dataset_loader
            },
            'datasets': {
                
                'data': dataset
            }        
        }
    
    
