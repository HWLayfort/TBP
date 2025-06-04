from DataLoader import TBPDataset, compute_scalers

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepONet(nn.Module):
    def __init__(
        self,
        branch_dim=21,
        trunk_dim=1,
        hidden_branch=128,
        hidden_trunk=128,
        p=128,
        output_dim=9,
        num_branch_layers=4,
        num_trunk_layers=4,
        dropout_rate=0.2,
        activation=nn.ReLU,
        output_activation=None
    ):
        super().__init__()
        self.p = p
        self.output_dim = output_dim

        # Branch Network
        branch_layers = []
        branch_layers.append(nn.Linear(branch_dim, hidden_branch))
        branch_layers.append(activation())
        for _ in range(num_branch_layers - 1):
            branch_layers.append(nn.Linear(hidden_branch, hidden_branch))
            branch_layers.append(activation())
            branch_layers.append(nn.Dropout(dropout_rate))
        branch_layers.append(nn.Dropout(dropout_rate))  # 추가
        branch_layers.append(nn.Linear(hidden_branch, p))
        self.branch_net = nn.Sequential(*branch_layers)

        # Trunk Network
        trunk_layers = []
        trunk_layers.append(nn.Linear(trunk_dim, hidden_trunk))
        trunk_layers.append(activation())
        for _ in range(num_trunk_layers - 1):
            trunk_layers.append(nn.Linear(hidden_trunk, hidden_trunk))
            trunk_layers.append(activation())
            trunk_layers.append(nn.Dropout(dropout_rate))
        trunk_layers.append(nn.Dropout(dropout_rate))  # 추가
        trunk_layers.append(nn.Linear(hidden_trunk, p))
        self.trunk_net = nn.Sequential(*trunk_layers)

        # Output map
        self.output_map = nn.Sequential(
            nn.Dropout(dropout_rate),            # output map 전에도 Dropout 추가 가능
            nn.Linear(p, output_dim)
        )
        self.output_activation = output_activation() if output_activation else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, branch_input, trunk_input):
        # branch_input: (B, branch_dim)
        # trunk_input: (B, N, trunk_dim)
        B, N, d_trunk = trunk_input.shape
        trunk_flat = trunk_input.reshape(B * N, d_trunk)

        # (선택사항) 정규화는 외부에서 처리하는 것이 안정적
        trunk_flat = (trunk_flat - trunk_flat.min()) / (trunk_flat.max() - trunk_flat.min() + 1e-8)

        branch_feat = self.branch_net(branch_input)               # (B, p)
        trunk_feat = self.trunk_net(trunk_flat)                  # (B*N, p)
        trunk_feat = trunk_feat.reshape(B, N, self.p)               # (B, N, p)

        interaction = trunk_feat * branch_feat.unsqueeze(1)      # (B, N, p)
        out = self.output_map(interaction)                       # (B, N, output_dim)
        out = self.output_activation(out)
        return out.permute(0, 2, 1)                              # (B, output_dim, N)

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

def train_deeponet(
    model, train_loader, val_loader, num_epochs=100, lr=1e-3, device='cuda',
    x_scaler=None, y_scaler=None,
    model_save_path='deeponet.pth', log_path='deeponet.log', early_stop_patience=20,
    phys_loss_weight=1e-7,
):
    print(f"Training DeepONet on device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    patience = 0
    if log_path is not None:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
    for epoch in range(num_epochs):
        model.train()
        train_loss, n = 0.0, 0
        for xb, yb, _, mb in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            t = xb[:, -1, :]
            tb = xb[:, 21:, :].permute(0, 2, 1)
            xb = x_scaler.transform(xb)
            yb = y_scaler.transform(yb)
            xb = xb[:, :21, :].mean(dim=2)
            optimizer.zero_grad()
            y_pred = model(xb, tb)
            loss_data = loss_fn(y_pred, yb)
            y_pred_phys = y_scaler.inverse_transform(y_pred)
            a_pred = compute_second_derivative(y_pred_phys, t)
            a_grav = compute_gravity_acceleration(y_pred_phys, mb)[:, :, 1:-1]  # <--- 수정
            loss_phys = ((a_pred - a_grav) ** 2).mean()
            loss_phys_scaled = loss_phys * (loss_data.detach() / (loss_phys.detach() + 1e-8))
            loss = loss_data + phys_loss_weight * loss_phys_scaled
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss /= n
        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb, _, _  in tqdm(val_loader, desc=f"Val {epoch+1}", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                tb = xb[:, 21:, :].permute(0, 2, 1)
                xb = x_scaler.transform(xb)
                yb = y_scaler.transform(yb)
                xb = xb[:, :21, :].mean(dim=2)
                y_pred = model(xb, tb)
                vloss = loss_fn(y_pred, yb)
                val_loss += vloss.item() * xb.size(0)
                n_val += xb.size(0)
        avg_val_loss = val_loss / n_val
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss},{avg_val_loss}\n")
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
            print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    print(f"Best validation loss: {best_val:.6f}")

def test_deeponet(model, test_loader, x_scaler, y_scaler, device='cuda', log_path='deeponet.log'):
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb, _, _ in tqdm(test_loader, desc="Test"):
            xb, yb = xb.to(device), yb.to(device)
            tb = xb[:, 21:, :].permute(0, 2, 1)
            xb = x_scaler.transform(xb)
            yb = y_scaler.transform(yb)
            xb = xb[:, :21, :].mean(dim=2)
            y_pred = model(xb, tb)
            loss = loss_fn(y_pred, yb)
            test_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    test_loss /= n
    print(f"[Test] Loss: {test_loss:.6f}")
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"test,{test_loss}\n")
    return test_loss

def run_deeponet_pipeline(train_loader, val_loader, test_loader, x_scaler, y_scaler, device='cuda'):
    model = DeepONet(
        branch_dim=21, trunk_dim=1, hidden_branch=128, hidden_trunk=128,
        p=128, output_dim=9, num_branch_layers=4, num_trunk_layers=4
    )
    train_deeponet(
        model, train_loader, val_loader, 
        num_epochs=500, lr=1e-3, device=device, 
        model_save_path=os.path.join(os.path.dirname(__file__), 'models', 'deeponet.pth'),
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'deeponet.log'),
        early_stop_patience=20,
        x_scaler=x_scaler, y_scaler=y_scaler
    )
    
    test_deeponet(
        model, test_loader, device=device, x_scaler=x_scaler, y_scaler=y_scaler, 
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'deeponet.log')
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
    
    run_deeponet_pipeline(
        train_loader, val_loader, test_loader, x_scaler, y_scaler, device=device
    )