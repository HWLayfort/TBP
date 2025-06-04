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
    # X: (B, C, N)
    sum_x, sum_x2, total = 0.0, 0.0, 0
    sum_y, sum_y2 = 0.0, 0.0

    for xb, yb, _, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        B, C, N = xb.shape
        x_flat = xb.permute(0, 2, 1).reshape(-1, C)  # (B*N, C)
        y_flat = yb.permute(0, 2, 1).reshape(-1, yb.shape[1])

        sum_x += x_flat.sum(dim=0)
        sum_x2 += (x_flat ** 2).sum(dim=0)

        sum_y += y_flat.sum(dim=0)
        sum_y2 += (y_flat ** 2).sum(dim=0)

        total += x_flat.shape[0]

    mean_x = sum_x / total
    std_x = (sum_x2 / total - mean_x**2).sqrt() + 1e-8

    mean_y = sum_y / total
    std_y = (sum_y2 / total - mean_y**2).sqrt() + 1e-8

    sx = StandardScaler()
    sy = StandardScaler()
    sx.mean_, sx.std_ = mean_x, std_x
    sy.mean_, sy.std_ = mean_y, std_y
    return sx, sy


class TBPDataset(Dataset):
    def __init__(self, filelist, preload=True):
        self.filelist = filelist
        self.data = []

        if preload:
            for fname in self.filelist:
                data = torch.load(fname)

                t = data["t"]
                r = data["r"].permute(0, 2, 1).reshape(len(t), -1).permute(1, 0)  # (T, 3, 3) → (9, T)
                v = data["v"].permute(0, 2, 1).reshape(len(t), -1).permute(1, 0)

                r0 = data["r0"].reshape(-1, 1).repeat(1, len(t))
                v0 = data["v0"].reshape(-1, 1).repeat(1, len(t))
                m = data["m"].reshape(-1, 1).repeat(1, len(t))
                t_mat = t.unsqueeze(0).repeat(1, 1)

                x = torch.cat([r0, v0, m, t_mat], dim=0)  # (22, T)
                self.data.append((x, r, v, data["m"]))
        else:
            self.data = None

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if self.data is not None:
            return self.data[idx]
        else:
            data = torch.load(self.filelist[idx])
            t = data["t"]
            r = data["r"].permute(0, 2, 1).reshape(len(t), -1).permute(1, 0)
            v = data["v"].permute(0, 2, 1).reshape(len(t), -1).permute(1, 0)

            r0 = data["r0"].reshape(-1, 1).repeat(1, len(t))
            v0 = data["v0"].reshape(-1, 1).repeat(1, len(t))
            m = data["m"].reshape(-1, 1).repeat(1, len(t))
            t_mat = t.unsqueeze(0).repeat(1, 1)
            x = torch.cat([r0, v0, m, t_mat], dim=0)
            return x, r, v, data["m"]


