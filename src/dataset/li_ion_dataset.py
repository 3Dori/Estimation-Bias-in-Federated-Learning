import torch
import pandas as pd
import numpy as np
from mat4py import loadmat
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

class MatDNNDataset(Dataset):
    def __init__(self, root, window_size):
        self.data = loadmat(root)
        self.window_size = window_size

        # Panasonic Ah capacity
        self.BATTERY_AH_CAPACITY = 2.9000

        # Construct dataframe from MATLAB data
        self.df = pd.DataFrame(self.data)
        self.df = self.df.T
        self.df = self.df.apply(lambda x : pd.Series(x[0]))
        self.df = self.df.applymap(lambda x : x[0])

        # Clean up unnecessary columns
        del self.df['Chamber_Temp_degC']
        del self.df['TimeStamp']
        del self.df['Time']
        del self.df['Power']
        del self.df['Wh']

        # Normalize
        # self.df = (self.df - self.df.mean())/self.df.std()

        # Add moving average columns
        # self.df['count'] = range(1,len(self.df)+1)
        # self.df['cum_Voltage'] = self.df['Voltage'].cumsum()
        # self.df['cum_Current'] = self.df['Current'].cumsum()

        # self.df['mov_Voltage'] = self.df['cum_Voltage'] / self.df['count']
        # self.df['mov_Current'] = self.df['cum_Current'] / self.df['count']

        # del self.df['cum_Voltage']
        # del self.df['cum_Current']
        # del self.df['count']

        self.df['rol_Voltage'] = self.df['Voltage'].rolling(window_size).mean()
        self.df['rol_Current'] = self.df['Current'].rolling(window_size).mean()

        # Add SOC column
        ah = self.df['Ah']
        self.df['SOC'] = 1 + (ah/self.BATTERY_AH_CAPACITY)
        del self.df['Ah']

        # Convert data to numpy
        self.data = self.df.to_numpy(dtype=np.float32)

        # Set values for dataset
        self.x = torch.from_numpy(self.data[:,:-1])
        self.y = torch.from_numpy(self.data[:,-1])

        # Reshape to match required label tensor shape
        self.y = self.y.reshape((len(self.y), 1))

    def __getitem__(self, idx):
        # Return window with the state of charge of last element in window (time series data + state of charge at the very end)
        return self.x[idx + self.window_size - 1, :], self.y[idx + self.window_size - 1]

    def __len__(self):
        # Subtract sequence length to ensure we don't sample out of bounds
        return len(self.data) - self.window_size + 1


class MatRNNDataset(Dataset):
    def __init__(self, root, sequence_length, window_size):
        # window_size is the sequence length
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.data = loadmat(root)

        # Panasonic Ah capacity
        self.BATTERY_AH_CAPACITY = 2.9000

        # Construct dataframe from MATLAB data
        self.df = pd.DataFrame(self.data)
        self.df = self.df.T
        self.df = self.df.apply(lambda x : pd.Series(x[0]))
        self.df = self.df.applymap(lambda x : x[0])

        # Clean up unnecessary columns
        del self.df['Chamber_Temp_degC']
        del self.df['TimeStamp']
        del self.df['Time']
        del self.df['Power']
        del self.df['Wh']

        # Normalize
        # self.df = (self.df - self.df.mean())/self.df.std()

        # Add moving average columns
        # self.df['count'] = range(1,len(self.df)+1)
        # self.df['cum_Voltage'] = self.df['Voltage'].cumsum()
        # self.df['cum_Current'] = self.df['Current'].cumsum()

        # self.df['mov_Voltage'] = self.df['cum_Voltage'] / self.df['count']
        # self.df['mov_Current'] = self.df['cum_Current'] / self.df['count']

        # del self.df['cum_Voltage']
        # del self.df['cum_Current']
        # del self.df['count']

        self.df['rol_Voltage'] = self.df['Voltage'].rolling(window_size).mean()
        self.df['rol_Current'] = self.df['Current'].rolling(window_size).mean()

        # Add SOC column
        ah = self.df['Ah']
        self.df['SOC'] = 1 + (ah/self.BATTERY_AH_CAPACITY)
        del self.df['Ah']

        # Convert data to numpy
        self.data = self.df.to_numpy(dtype=np.float32)

        # Set values for dataset
        self.x = torch.from_numpy(self.data[:,:-1])
        self.y = torch.from_numpy(self.data[:,-1])

        # Reshape to match required label tensor shape
        self.y = self.y.reshape((len(self.y), 1))

    def __getitem__(self, idx):
        # Return window with the state of charge of last element in window (time series data + state of charge at the very end)
        start = idx + self.window_size
        end = idx + self.window_size + self.sequence_length #Exclusive
        return self.x[start:end, :], self.y[end - 1]

    def __len__(self):
        # Subtract sequence length to ensure we don't sample out of bounds
        return len(self.data) - self.sequence_length - self.window_size + 1

class OCVDataset(Dataset):
    def __init__(self, root, sheet_name, n_inp):
        self.df = pd.read_excel(root, sheet_name=sheet_name)
        self.window_size = window_size

        del self.df['Date_Time']

        self.data = self.df.to_numpy(dtype=np.float32)
        self.x = torch.from_numpy(self.data[:,:n_inp])
        self.y = torch.from_numpy(self.data[:,n_inp:])

    def __getitem__(self, idx):
        return self.x[idx : idx + self.window_size, :], self.y[idx,:]

    def __len__(self):
        return len(self.data) - self.window_size


if(__name__ == "__main__"):
    pass


def get_li_ion_dataloader(train_data_path="data/Panasonic 18650PF Data/0degC/Drive cycles/",
                          train_files=[
                              "05-30-17_12.56 0degC_Cycle_1_Pan18650PF.mat",
                              "05-30-17_20.16 0degC_Cycle_2_Pan18650PF.mat",
                              "06-01-17_15.36 0degC_Cycle_3_Pan18650PF.mat",
                              "06-01-17_22.03 0degC_Cycle_4_Pan18650PF.mat"
                          ],
                          test_data_path="data/Panasonic 18650PF Data/0degC/Drive cycles/06-02-17_10.43 0degC_HWFET_Pan18650PF.mat",
                          window_size=20, batch_size=None, test_batch_size=1000):
    import os.path
    train_datasets = [MatDNNDataset(os.path.join(train_data_path, f), window_size) for f in train_files]
    test_dataset = MatDNNDataset(test_data_path, window_size)

    train_loaders = [DataLoader(dataset=d, batch_size=batch_size, shuffle=True) for d in train_datasets]
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loaders, test_loader
