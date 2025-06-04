from DataLoader import TBPDataset, compute_scalers

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# =======================
# SpectralConv1d/FNO1d 정의 (논문 구조)
# =======================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalFourierEmbedding(nn.Module):
    def __init__(self, n_steps, embed_dim=32, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn((1, embed_dim)) * scale, requires_grad=False)
        t_lin = torch.linspace(0, 1, n_steps).unsqueeze(1)  # (n_steps, 1)
        x_proj = 2 * np.pi * t_lin @ self.B  # (n_steps, embed_dim)
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)  # (n_steps, 2*embed_dim)
        self.register_buffer('t_embedding', emb)

    def forward(self, batch_size):
        return self.t_embedding[None, :, :].expand(batch_size, -1, -1)  # (B, N, D)

class SpectralConv1dTFNO(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        scale = 0.02
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, N//2+1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights)
        x = torch.fft.irfft(out_ft, n=N, dim=-1)
        return x

class ChannelMixing(nn.Module):
    def __init__(self, width, hidden_mul=2, dropout=0.0, activation=nn.GELU):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(width, hidden_mul * width, 1),
            activation(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_mul * width, width, 1)
        )

    def forward(self, x):
        return self.layer(x)

class TFNO(nn.Module):
    def __init__(
        self, t_steps, input_dim=22, embed_dim=32, width=64,
        modes=32, depth=4, output_dim=9, activation=nn.GELU, dropout=0.2
    ):
        super().__init__()
        self.temb = TemporalFourierEmbedding(t_steps, embed_dim)
        self.fc0 = nn.Linear(input_dim + 2 * embed_dim, width)

        self.spectral_layers = nn.ModuleList([
            SpectralConv1dTFNO(width, width, modes) for _ in range(depth)
        ])
        self.channel_mixings = nn.ModuleList([
            ChannelMixing(width, hidden_mul=2, dropout=dropout, activation=activation) for _ in range(depth)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(width) for _ in range(depth)
        ])

        self.fc1 = nn.Linear(width, 128)
        self.dropout = nn.Dropout(dropout)  # <-- 추가된 dropout
        self.fc2 = nn.Linear(128, output_dim)
        self.activation = activation()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, input_dim, N = x.shape
        t_emb = self.temb(B)                          # (B, N, 2*embed)
        x_in = torch.cat([x.permute(0, 2, 1), t_emb], dim=-1)  # (B, N, input_dim+embed)
        x = self.fc0(x_in)                            # (B, N, width)
        x = x.permute(0, 2, 1)                        # (B, width, N)

        for conv, cmix, norm in zip(self.spectral_layers, self.channel_mixings, self.norms):
            x1 = conv(x)
            x2 = cmix(x)
            x = self.activation(x1 + x2)
            x = norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # post-norm

        x = x.permute(0, 2, 1)  # (B, N, width)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)     # <-- fc1과 fc2 사이에 dropout
        x = self.fc2(x)         # (B, N, output_dim)
        return x.permute(0, 2, 1)  # (B, output_dim, N)

def train_tfno(
    model, train_loader, val_loader, num_epochs=100, lr=1e-3, device='cuda',
    x_scaler=None, y_scaler=None,
    model_save_path='tfno.pth', log_path='tfno.log', early_stop_patience=20
):
    print(f"Training TFNO on device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    patience = 0
    if log_path is not None:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb, _, _ in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            xb, yb = x_scaler.transform(xb.to(device)), y_scaler.transform(yb.to(device))
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        avg_loss = total_loss / n
        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb, _, _ in tqdm(val_loader, desc=f"Val {epoch+1}", leave=False):
                xb, yb = x_scaler.transform(xb.to(device)), y_scaler.transform(yb.to(device))
                y_pred = model(xb)
                vloss = loss_fn(y_pred, yb)
                val_loss += vloss.item() * xb.size(0)
                n_val += xb.size(0)
        avg_val_loss = val_loss / n_val
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(f"{epoch},{avg_loss},{avg_val_loss}\n")
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}: no improvement in {early_stop_patience} epochs.")
                break
        if epoch % 10 == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    print(f"Best validation loss: {best_val:.6f}")

def test_tfno(model, test_loader, x_scaler, y_scaler, device='cuda', log_path='tfno.log'):
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb, _, _ in tqdm(test_loader, desc="Test"):
            xb, yb = x_scaler.transform(xb.to(device)), y_scaler.transform(yb.to(device))
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            test_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    test_loss /= n
    print(f"[Test] Loss: {test_loss:.6f}")
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"test,{test_loss}\n")
    return test_loss

# =======================
# 전체 파이프라인
# =======================

def run_tfno_pipeline(train_loader, val_loader, test_loader, x_scaler, y_scaler, device='cuda'):
    # n_steps는 시계열 길이(파일마다 동일해야 함, 예시: 2000)
    n_steps = 100
    print(f"Number of time steps: {n_steps}")
    model = TFNO(
        t_steps=100, input_dim=22, embed_dim=32, width=64, modes=32,
        depth=4, output_dim=9
    )
    train_tfno(
        model, train_loader, val_loader, x_scaler=x_scaler, y_scaler=y_scaler,
        num_epochs=10000, lr=1e-3, device=device, 
        model_save_path=os.path.join(os.path.dirname(__file__), 'models', 'tfno.pth'),
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'tfno.log'),
        early_stop_patience=20
    )
    
    test_tfno(
        model, test_loader, x_scaler=x_scaler, y_scaler=y_scaler, device=device,
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'tfno.log')
    )
    
if __name__ == "__main__":
    train_dir = os.path.join(os.path.dirname(__file__), "data", "train")
    train_file_list = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir) if fname.endswith('.pt')]
    train_dataset = TBPDataset(train_file_list[:10000], preload=False)  # For testing, limit to 100 files
    train_ds, val_ds, _ = random_split(train_dataset, [0.8, 0.2, 0], generator=torch.Generator().manual_seed(42))
    print(f"Loaded train dataset with {len(train_dataset)} samples.")
    
    test_dir = os.path.join(os.path.dirname(__file__), "data", "test")
    test_file_list = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith('.pt')]
    test_ds = TBPDataset(test_file_list)
    print(f"Loaded test dataset with {len(test_ds)} samples.")
    
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    print (f"Created DataLoaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_scaler, y_scaler = compute_scalers(train_loader, device)
    print("Computed data scalers for training data.")
    run_tfno_pipeline(
        train_loader, val_loader, test_loader, x_scaler, y_scaler, device=device
    )