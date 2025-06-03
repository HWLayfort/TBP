from DataLoader import TBPDataset, compute_scalers
from PINN import train_pinn, test_pinn

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        scale = 1 / (in_channels * out_channels)
        # real + imag 저장 후 view_as_complex 사용
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, 2)  # 마지막 dim=2 → real + imag
        )

    def compl_mul1d(self, input, weights):
        # input: (B, in_c, modes) complex
        # weights: (in_c, out_c, modes) complex
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, N//2 + 1), complex
        weights_cplx = torch.view_as_complex(self.weights)  # (in_c, out_c, modes), complex

        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], weights_cplx)
        x = torch.fft.irfft(out_ft, n=N, dim=-1)  # (B, out_c, N)
        return x

class HybridPINN(nn.Module):
    def __init__(self, input_dim=22, width=64, modes=16, hidden_dim=128, depth=3, output_dim=9, dropout_rate=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.modes = modes

        self.fc_in = nn.Linear(input_dim, width)  # 입력 선형 변환
        self.spectral = SpectralConv1d(width, width, modes)  # SpectralConv1d로 특성 추출
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

        # 일반 PINN 구조로 넘어감
        self.fc_blocks = nn.ModuleList()
        for _ in range(depth):
            self.fc_blocks.append(nn.Linear(width, hidden_dim))
            self.fc_blocks.append(nn.Tanh())
            self.fc_blocks.append(nn.Dropout(dropout_rate))
            width = hidden_dim  # 이후 hidden_dim으로 계속 진행

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x: (B, D, N)
        B, D, N = x.shape
        x = x.permute(0, 2, 1).reshape(B * N, D)        # (B*N, D)
        x = self.fc_in(x)                               # (B*N, width)
        x = x.view(B, N, -1).permute(0, 2, 1)           # (B, width, N)
        x = self.spectral(x)                            # (B, width, N)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1).reshape(B * N, -1)       # (B*N, width)

        for layer in self.fc_blocks:
            x = layer(x)

        x = self.fc_out(x)
        return x.view(B, N, -1).permute(0, 2, 1)        # (B, output_dim, N)

def run_hybrid_pinn_pipeline(train_loader, val_loader, test_loader, x_scaler, y_scaler, device='cuda'):
    model = HybridPINN(input_dim=22, width=64, modes=16, hidden_dim=128, depth=3, output_dim=9)
    train_pinn(model, train_loader, val_loader, num_epochs=500, device=device, 
               x_scaler=x_scaler, y_scaler=y_scaler,
               model_save_path=os.path.join(os.path.dirname(__file__), 'models', 'hybrid_pinn.pth'),
               log_path=os.path.join(os.path.dirname(__file__), 'logs', 'hybrid_pinn.log'),
               early_stop_patience=20,
               phys_loss_weight=0.5)
    
    test_pinn(model, test_loader, device=device, x_scaler=x_scaler, y_scaler=y_scaler,
              log_path=os.path.join(os.path.dirname(__file__), 'logs', 'hybrid_pinn.log'))

if __name__ == "__main__":
    train_dir = os.path.join(os.path.dirname(__file__), "data", "train")
    train_file_list = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir) if fname.endswith('.csv')]
    train_dataset = TBPDataset(train_file_list)  # For testing, limit to 100 files
    train_ds, val_ds, _ = random_split(train_dataset, [0.8, 0.2, 0], generator=torch.Generator().manual_seed(42))
    print(f"Loaded train dataset with {len(train_dataset)} samples.")
    
    test_dir = os.path.join(os.path.dirname(__file__), "data", "test")
    test_file_list = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith('.csv')]
    test_ds = TBPDataset(test_file_list)
    print(f"Loaded test dataset with {len(test_ds)} samples.")
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_scaler, y_scaler = compute_scalers(train_loader, device)
    print("Computed data scalers for training data.")
    run_hybrid_pinn_pipeline(
        train_loader, val_loader, test_loader, 
        x_scaler=x_scaler, y_scaler=y_scaler, device=device
    )
