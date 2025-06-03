from DataLoader import TBPDataset, compute_scalers

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# =======================
# SpectralConv1d/FNO1d 정의 (논문 구조)
# =======================

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


class FNO(nn.Module):
    def __init__(
        self, modes=16, width=64, input_dim=1, output_dim=1, depth=4,
        hidden_fc=128, activation=nn.GELU, use_layernorm=False,
        dropout_rate=0.2
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.depth = depth
        self.use_layernorm = use_layernorm
        self.activation = activation()

        self.fc0 = nn.Linear(input_dim, width)
        self.convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)]) if use_layernorm else None

        self.fc1 = nn.Linear(width, hidden_fc)
        self.dropout = nn.Dropout(dropout_rate)  # dropout layer 추가
        self.fc2 = nn.Linear(hidden_fc, output_dim)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C_in, N = x.shape
        x = x.permute(0, 2, 1)     # (B, N, input_dim)
        x = self.fc0(x)            # (B, N, width)
        x = x.permute(0, 2, 1)     # (B, width, N)

        for i in range(self.depth):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = self.activation(x1 + x2)
            x = self.dropout(x)  # dropout after each spectral block

            if self.use_layernorm:
                x = x.permute(0, 2, 1)
                x = self.norms[i](x)
                x = x.permute(0, 2, 1)

        x = x.permute(0, 2, 1)         # (B, N, width)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)            # dropout before final output
        x = self.fc2(x)
        return x.permute(0, 2, 1)      # (B, output_dim, N)

# =======================
# FNO 학습 및 평가
# =======================

def train_fno(
    model, train_loader, val_loader, x_scaler=None, y_scaler=None,
    num_epochs=100, lr=1e-3, device='cuda', 
    model_save_path='fno_best.pth', 
    log_path='fno.log', 
    early_stop_patience=20
):
    print(f"Training FNO on device: {device}")
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
            xb, yb = xb.to(device), yb.to(device)
            xb, yb = x_scaler.transform(xb), y_scaler.transform(yb)
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
                xb, yb = xb.to(device), yb.to(device)
                xb, yb = x_scaler.transform(xb), y_scaler.transform(yb)
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

def test_fno(model, test_loader, x_scaler, y_scaler, device='cuda', log_path='fno.log'):
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

def run_fno_pipeline(train_loader, val_loader, test_loader, x_scaler, y_scaler, device='cuda'):
    model = FNO(modes=32, width=64, input_dim=22, output_dim=9, depth=4)
    train_fno(
        model, train_loader, val_loader, x_scaler=x_scaler, y_scaler=y_scaler,
        num_epochs=500, lr=1e-3, device=device, 
        model_save_path=os.path.join(os.path.dirname(__file__), 'models', 'fno.pth'),
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'fno.log'),
        early_stop_patience=20
    )
    
    test_fno(
        model, test_loader, x_scaler=x_scaler, y_scaler=y_scaler, device=device,
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'fno.log')
    )

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
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_scaler, y_scaler = compute_scalers(train_loader, device)
    print("Computed data scalers for training data.")

    run_fno_pipeline(
        train_loader, val_loader, test_loader, x_scaler, y_scaler, device=device
    )