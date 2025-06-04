from DataLoader import TBPDataset, compute_scalers

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# =====================
# PINN Model
# =====================
class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, depth=4, output_dim=1, dropout_rate=0.0):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(dropout_rate))  # 첫 층 뒤에 드롭아웃

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))  # 각 은닉층 뒤에 드롭아웃

        layers.append(nn.Linear(hidden_dim, output_dim))  # 출력층에는 드롭아웃 적용하지 않음
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, D, N)
        B, D, N = x.shape
        x = x.permute(0, 2, 1).reshape(B * N, D)  # (B*N, D)
        out = self.net(x)  # (B*N, output_dim)
        return out.view(B, N, -1).permute(0, 2, 1)  # (B, output_dim, N)

# =====================
# Physics Components
# =====================
def compute_second_derivative(y, t):
    # y의 시점 축은 N, 가운데만 사용하면 N-2
    y_prev = y[:, :, :-2]    # y_{i-1}
    y_curr = y[:, :, 1:-1]   # y_i
    y_next = y[:, :, 2:]     # y_{i+1}

    # t는 (B, N), 아래와 같이 맞춰야 한다
    t_prev = t[:, :-2]
    t_curr = t[:, 1:-1]
    t_next = t[:, 2:]

    dt_forward = t_next - t_curr  # Δt+1
    dt_backward = t_curr - t_prev  # Δt
    dt = 0.5 * (dt_forward + dt_backward)  # (B, N-2)
    dt = dt.unsqueeze(1)  # (B, 1, N-2)

    return (y_next - 2 * y_curr + y_prev) / (dt ** 2)

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

def get_phys_weight(epoch, warmup_end=10, peak_epoch=50, init_weight=1e-7, decay_rate=0.98):
    if epoch < warmup_end:
        return 0.0
    elif warmup_end <= epoch < peak_epoch:
        # 선형 증가
        return init_weight * ((epoch - warmup_end) / (peak_epoch - warmup_end))
    else:
        # 지수 감소
        return init_weight * (decay_rate ** (epoch - peak_epoch))

# =====================
# Training Loop
# =====================
def train_pinn(
    model, train_loader, val_loader, num_epochs=100, lr=1e-3, device='cuda',
    x_scaler=None, y_scaler=None,
    model_save_path='pinn.pth', log_path='pinn.log', early_stop_patience=50,
    phys_loss_weight_init=1, phys_weight_decay=0.98,
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    patience = 0
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_phys,val_loss,val_phys\n")
    
    for epoch in range(num_epochs):
        model.train()
        phys_loss_weight = get_phys_weight(
            epoch,
            warmup_end=10,        # 예: 10 에폭까지는 무시
            peak_epoch=50,        # 50 에폭에 최대 weight 도달
            init_weight=phys_loss_weight_init,
            decay_rate=phys_weight_decay
        )
        train_loss, train_phys, n = 0.0, 0.0, 0
        for xb, yb, _, mb in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            t = xb[:, -1, :]
            x_sclaed, y_sclaed = x_scaler.transform(xb), y_scaler.transform(yb)
            optimizer.zero_grad()
            y_pred = model(x_sclaed)
            loss_data = loss_fn(y_pred, y_sclaed)
            y_pred_phys = y_scaler.inverse_transform(y_pred)
            a_pred = compute_second_derivative(y_pred_phys, t)
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
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                t = xb[:, -1, :]
                x_sclaed, y_sclaed = x_scaler.transform(xb), y_scaler.transform(yb)
                y_pred = model(x_sclaed)
                loss_data = loss_fn(y_pred, y_sclaed)
                y_pred_phys = y_scaler.inverse_transform(y_pred)
                a_pred = compute_second_derivative(y_pred_phys, t)
                a_grav = compute_gravity_acceleration(y_pred_phys, mb)[:, :, 1:-1]  # <--- 수정
                loss_phys = ((a_pred - a_grav) ** 2).mean()
                val_loss += loss_data.item() * xb.size(0)
                val_phys += loss_phys.item() * xb.size(0)
                nv += xb.size(0)
        val_loss /= nv
        val_phys /= nv

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

# =====================
# Test Loop
# =====================
def test_pinn(model, test_loader, x_scaler, y_scaler, device='cuda', log_path='pinn.log'):
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

# =====================
# Run Pipeline
# =====================
def run_pinn_pipeline(train_loader, val_loader, test_loader, x_scaler, y_scaler, device='cuda'):
    model = PINN(input_dim=22, hidden_dim=128, depth=4, output_dim=9)
    train_pinn(model, train_loader, val_loader, num_epochs=500, device=device, 
               x_scaler=x_scaler, y_scaler=y_scaler,
                model_save_path=os.path.join(os.path.dirname(__file__), 'models', 'pinn.pth'),
                log_path=os.path.join(os.path.dirname(__file__), 'logs', 'pinn.log'),
               early_stop_patience=1000)
    test_pinn(
        model, test_loader, device=device, x_scaler=x_scaler, y_scaler=y_scaler,
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'pinn.log')
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
    run_pinn_pipeline(
        train_loader, val_loader, test_loader, 
        x_scaler=x_scaler, y_scaler=y_scaler, device=device
    )
