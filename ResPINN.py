from DataLoader import TBPDataset, compute_scalers

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ================================
# ResPINN Trajectory Model
# ================================
class FourierEmbedding(nn.Module):
    def __init__(self, embed_dim=32, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn((1, embed_dim)) * scale, requires_grad=False)

    def forward(self, t):  # t: (B, N, 1)
        x_proj = 2 * np.pi * t @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (B, N, 2*embed_dim)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.Tanh()

    def forward(self, x):  # (B*N, dim)
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return residual + out

class ResPINN(nn.Module):
    def __init__(self, input_dim=22, embed_dim=32, hidden_dim=128, depth=6, output_dim=9, dropout_rate=0.2):
        super().__init__()
        self.fourier = FourierEmbedding(embed_dim=embed_dim)
        total_input_dim = input_dim + 2 * embed_dim

        self.input_layer = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate=dropout_rate) for _ in range(depth)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_cond, t):  
        # x_cond: (B, 21), t: (B, N, 1)
        B, N, _ = t.shape
        t_feat = self.fourier(t)  # (B, N, 2*embed_dim)
        x_repeated = x_cond.unsqueeze(1).expand(-1, N, -1)  # (B, N, 21)
        x_input = torch.cat([x_repeated, t_feat], dim=-1)   # (B, N, 21+2*embed_dim)
        x_input = x_input.view(B * N, -1)                   # (B*N, D)

        x = self.input_layer(x_input)                       # (B*N, hidden)
        x = self.res_blocks(x)                              # (B*N, hidden)
        x = self.output_layer(x)                            # (B*N, output_dim)
        return x.view(B, N, -1).permute(0, 2, 1)             # (B, output_dim, N)

# =====================
# Physics Components
# =====================
def compute_second_derivative(y, dt):
    dt = dt.view(-1, 1, 1)  # reshape to (B, 1, 1) for broadcasting
    return (y[:, :, 2:] - 2 * y[:, :, 1:-1] + y[:, :, :-2]) / (dt ** 2)


def compute_gravity_acceleration(r, masses, G=1.0, eps=1e-8):
    B, _, N = r.shape
    r = r.view(B, 3, 3, N)
    a = torch.zeros_like(r)
    for i in range(3):
        for j in range(3):
            if i != j:
                rij = r[:, j, :, :] - r[:, i, :, :]
                dist = torch.norm(rij, dim=1, keepdim=True) + eps
                a[:, i, :, :] += G * masses[:, j].unsqueeze(-1).unsqueeze(-1) * rij / (dist ** 3)
    return a.view(B, 9, N)

# ================================
# Training Pipeline
# ================================
def train_respinn(
    model, train_loader, val_loader, num_epochs=500, lr=1e-3,
    device='cuda', model_save_path='respinn.pth',
    x_scaler=None, y_scaler=None,
    log_path='respinn.log', early_stop_patience=100,
    phys_loss_weight=1e-7
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    patience = 0

    if log_path:
        with open(log_path, 'w') as f:
            f.write("epoch,train_loss,train_phys,val_loss,val_phys\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_phys, n = 0.0, 0.0, 0
        for xb, yb, _, mb in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            xb, yb, mb = x_scaler.transform(xb.to(device)), y_scaler.transform(yb.to(device)), mb.to(device)    
            x_cond = xb[:, :21, 0]                 # (B, 21)
            t = xb[:, 21:, :].permute(0, 2, 1)     # (B, N, 1)
            optimizer.zero_grad()
            y_pred = model(x_cond, t)
            loss_data = loss_fn(y_pred, yb)
            dt = xb[:, -1, 1] - xb[:, -1, 0]
            y_pred_phys = y_scaler.inverse_transform(y_pred)
            a_pred = compute_second_derivative(y_pred_phys, dt)
            a_grav = compute_gravity_acceleration(y_pred_phys, mb)[:, :, 1:-1]  # <--- 수정
            loss_phys = ((a_pred - a_grav) ** 2).mean()
            loss = loss_data + phys_loss_weight * loss_phys
            loss.backward()
            optimizer.step()
            train_loss += loss_data.item() * xb.size(0)
            train_phys += loss_phys.item() * xb.size(0)
            n += xb.size(0)
        train_loss /= n
        train_phys /= n

        # Validation
        model.eval()
        val_loss, val_phys, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb, _, mb in tqdm(val_loader, desc=f"Val {epoch+1}", leave=False):
                xb, yb, mb = x_scaler.transform(xb.to(device)), y_scaler.transform(yb.to(device)), mb.to(device)    
                x_cond = xb[:, :21, 0]                 # (B, 21)
                t = xb[:, 21:, :].permute(0, 2, 1)     # (B, N, 1)
                y_pred = model(x_cond, t)
                loss_data = loss_fn(y_pred, yb)
                dt = xb[:, -1, 1] - xb[:, -1, 0]
                y_pred_phys = y_scaler.inverse_transform(y_pred)
                a_pred = compute_second_derivative(y_pred_phys, dt)
                a_grav = compute_gravity_acceleration(y_pred_phys, mb)[:, :, 1:-1]  # <--- 수정
                loss_phys = ((a_pred - a_grav) ** 2).mean()
                val_loss += loss_data.item() * xb.size(0)
                val_phys += loss_phys.item() * xb.size(0)
                nv += xb.size(0)
        val_loss /= nv
        val_phys /= nv

        if log_path:
            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss},{train_phys},{val_loss},{val_phys}\n")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if epoch % 10 == 0:
            print(f"[Epoch {epoch+1}] Train: {train_loss:.6f}, Phys: {train_phys:.6f} | Val: {val_loss:.6f}, Phys: {val_phys:.6f}")

    print(f"Best validation loss: {best_val:.6f}")

def test_respinn(model, test_loader, x_scaler, y_scaler, device='cuda', log_path='respinn.log'):
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb, _, mb in tqdm(test_loader, desc="Test"):
            xb, yb, mb = x_scaler.transform(xb.to(device)), y_scaler.transform(yb.to(device)), mb.to(device)    
            x_cond = xb[:, :21, 0]                 # (B, 21)
            t = xb[:, 21:, :].permute(0, 2, 1)     # (B, N, 1)
            y_pred = model(x_cond, t)
            loss = loss_fn(y_pred, yb)
            test_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    test_loss /= n
    print(f"[Test] Loss: {test_loss:.6f}")
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"test,{test_loss}\n")
    return test_loss

# ================================
# Run Pipeline
# ================================
def run_respinn_pipeline(train_loader, val_loader, test_loader, x_scaler, y_scaler, device='cuda'):
    model = ResPINN(input_dim=21, embed_dim=32, hidden_dim=128, depth=6, output_dim=9)
    train_respinn(
        model, train_loader, val_loader, num_epochs=500, 
        device=device, 
        model_save_path=os.path.join(os.path.dirname(__file__), 'models', 'respinn.pth'),
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'respinn.log'),
        x_scaler=x_scaler, y_scaler=y_scaler)
    test_respinn(
        model, test_loader, device=device, x_scaler=x_scaler, y_scaler=y_scaler,
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'respinn.log')
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
    run_respinn_pipeline(
        train_loader, val_loader, test_loader, 
        x_scaler=x_scaler, y_scaler=y_scaler, device=device
    )
