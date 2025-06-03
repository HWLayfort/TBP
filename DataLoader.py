import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class StandardScaler:
    def fit(self, X):
        # X: (B, C, N)
        X = X.permute(0, 2, 1).reshape(-1, X.shape[1])   # (B*N, C)
        self.mean_ = torch.mean(X, dim=0)
        self.std_ = torch.std(X, dim=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_.view(1, -1, 1)) / self.std_.view(1, -1, 1)

    def inverse_transform(self, X):
        return X * self.std_.view(1, -1, 1) + self.mean_.view(1, -1, 1)
    
def compute_scalers(loader, device):
    xs, ys = [], []
    for xb, yb, _, _ in loader:
        xs.append(xb)
        ys.append(yb)
    x_all = torch.cat(xs, dim=0).to(device)
    y_all = torch.cat(ys, dim=0).to(device)
    sx, sy = StandardScaler().fit(x_all), StandardScaler().fit(y_all)
    return sx, sy

class TBPDataset(Dataset):
    def __init__(self, filelist):
        self.data = []
        for fname in filelist:
            base = os.path.basename(fname).split('_')[0]
            masses = np.array([float(x) for x in base.replace('m', '').split('-')])
            df = pd.read_csv(fname, header=None)
            t = df.iloc[:, 0].values
            r = df.iloc[:, 1:10].values.T
            v = df.iloc[:, 10:19].values.T
            r0 = df.iloc[0, 1:10].values
            v0 = df.iloc[0, 10:19].values
            N = len(t)
            r0_mat = np.tile(r0[:, None], (1, N))
            v0_mat = np.tile(v0[:, None], (1, N))
            m_mat = np.tile(masses[:, None], (1, N))
            t_mat = t[None, :]
            x = np.concatenate([r0_mat, v0_mat, m_mat, t_mat], axis=0)
            self.data.append((x, r, v, masses))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, r, v, m = self.data[idx]
        return (
            torch.tensor(x, dtype=torch.float32),  # (22, N)
            torch.tensor(r, dtype=torch.float32),  # (9, N)
            torch.tensor(v, dtype=torch.float32),  # (9, N)
            torch.tensor(m, dtype=torch.float32),  # (3,)
        )
