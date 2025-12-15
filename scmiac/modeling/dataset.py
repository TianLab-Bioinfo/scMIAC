import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler



class scDataset(Dataset):
    """
    A PyTorch dataset for handling single-cell data with optional scaling and labels.
    """
    def __init__(self, data, labels=None, scaler=None):
        self.data = data
        self.labels = labels
        
        self.scaler = scaler
        if self.scaler is not None:
            self.data = self.scaler.fit_transform(self.data)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]




class MultiOmicDataset(Dataset):
    """
    A PyTorch dataset for integrating RNA and ATAC data, ensuring batch consistency using modulo indexing.
    """
    def __init__(self, rna_arr, atac_arr):
        scaler_rna = StandardScaler()
        self.rna_data = scaler_rna.fit_transform(rna_arr)
        scaler_atac = StandardScaler()
        self.atac_data = scaler_atac.fit_transform(atac_arr)
        self.size = max(len(self.rna_data), len(self.atac_data))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Use the modulo operation to cycle through samples from the shorter dataset, 
        # ensuring that each batch contains both RNA and ATAC data (to do).

        rna_idx = idx % len(self.rna_data)
        atac_idx = idx % len(self.atac_data)

        rna_sample = self.rna_data[rna_idx]
        atac_sample = self.atac_data[atac_idx]

        return {
            "rna": torch.tensor(rna_sample, dtype=torch.float32),
            "atac": torch.tensor(atac_sample, dtype=torch.float32)
        }
    




class AnchorCellsDataset(Dataset):
    """
    A PyTorch dataset for paired anchor cells, mapping RNA and ATAC data based on an anchor DataFrame.
    """

    def __init__(self, rna_arr, atac_arr, anchor_df: pd.DataFrame):
        scaler_rna = StandardScaler()
        self.rna_data = scaler_rna.fit_transform(rna_arr)
        scaler_atac = StandardScaler()
        self.atac_data = scaler_atac.fit_transform(atac_arr)
        self.anchor_df = anchor_df.reset_index(drop=True)

    def __len__(self):
        return len(self.anchor_df)

    def __getitem__(self, idx):
        rna_idx = self.anchor_df.iloc[idx]['x1']
        atac_idx = self.anchor_df.iloc[idx]['x2']
        
        rna_sample = self.rna_data[rna_idx]
        atac_sample = self.atac_data[atac_idx]
        ct = self.anchor_df.iloc[idx]['x1_ct']
        return {
            "rna_anchor": torch.tensor(rna_sample, dtype=torch.float32),
            "atac_anchor": torch.tensor(atac_sample, dtype=torch.float32),
            "ct":ct
        }
    





# triple_omics
class MultiOmicDataset3(Dataset):
    """
    A PyTorch dataset for integrating RNA, ATAC, and methylation data, using modulo indexing to balance batch composition.
    """
    def __init__(self, rna_arr, atac_arr, meth_arr):
        scaler_rna = StandardScaler()
        self.rna_data = scaler_rna.fit_transform(rna_arr)
        
        scaler_atac = StandardScaler()
        self.atac_data = scaler_atac.fit_transform(atac_arr)
        
        scaler_meth = StandardScaler()
        self.meth_data = scaler_meth.fit_transform(meth_arr)

        self.size = max(len(self.rna_data), len(self.atac_data), len(self.meth_data))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        rna_idx = idx % len(self.rna_data)
        atac_idx = idx % len(self.atac_data)
        meth_idx = idx % len(self.meth_data)

        rna_sample = self.rna_data[rna_idx]
        atac_sample = self.atac_data[atac_idx]
        meth_sample = self.meth_data[meth_idx]

        return {
            "rna": torch.tensor(rna_sample, dtype=torch.float32),
            "atac": torch.tensor(atac_sample, dtype=torch.float32),
            "meth": torch.tensor(meth_sample, dtype=torch.float32)
        }
