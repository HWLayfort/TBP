# =====================
# OL-PINN Pipeline
# =====================
# DeepONet 사전학습 → 사전학습된 feature를 PINN 입력에 결합하여 residual 보정 학습

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from DataLoader import TBPDataset, compute_scalers
from DeepONet import DeepONet
from PINN import PINN, compute_second_derivative, compute_gravity_acceleration

class OL_PINN(nn.Module):
    def __init__(self, deeponet_ckpt, input_dim=22, hidden_dim=128, depth=4, output_dim=9, device='cuda'):
        super().__init__()
        self.device = device
        self.deeponet = DeepONet(branch_dim=21, trunk_dim=1, output_dim=output_dim).to(device)
        self.deeponet.load_state_dict(torch.load(deeponet_ckpt, map_location=device))
        self.deeponet.eval()  # freeze
        for param in self.deeponet.parameters():
            param.requires_grad = False

        # PINN은 DeepONet의 출력을 보정함
        self.pinn = PINN(input_dim=input_dim, hidden_dim=hidden_dim, depth=depth, output_dim=output_dim).to(device)

    def forward(self, xb):
        # xb: (B, D, N) → 3체 기준 D=22, N=시점 수
        with torch.no_grad():
            B, D, N = xb.shape
            branch_input = xb[:, :21, :].mean(dim=2)   # (B, 21)
            trunk_input = xb[:, 21:, :].permute(0, 2, 1)  # (B, N, 1)
            base_pred = self.deeponet(branch_input, trunk_input)  # (B, 9, N)

        residual = self.pinn(xb)  # (B, 9, N)
        return base_pred + residual

def train_ol_pinn(model, train_loader, val_loader, x_scaler, y_scaler, 
                  model_save_path, log_path, num_epochs=500, lr=1e-3, early_stop_patience=20,
                  device='cuda', phys_loss_weight=3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.pinn.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    patience = 0

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_phys,val_loss,val_phys\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_phys, n = 0.0, 0.0, 0
        for xb, yb, _, mb in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            t = xb[:, -1, :]
            x_scaled, y_scaled = x_scaler.transform(xb), y_scaler.transform(yb)
            optimizer.zero_grad()
            y_pred = model(x_scaled)
            loss_data = loss_fn(y_pred, y_scaled)
            y_pred_phys = y_scaler.inverse_transform(y_pred)
            a_pred = compute_second_derivative(y_pred_phys, t)
            a_grav = compute_gravity_acceleration(y_pred_phys, mb)[:, :, 1:-1]
            loss_phys = ((a_pred - a_grav) ** 2).mean()
            loss_phys_scaled = loss_phys * (loss_data.detach() / (loss_phys.detach() + 1e-8))
            loss = loss_data + phys_loss_weight * loss_phys_scaled
            loss.backward()
            optimizer.step()
            train_loss += loss_data.item() * xb.size(0)
            train_phys += loss_phys.item() * xb.size(0)
            n += xb.size(0)
        train_loss /= n
        train_phys /= n

        model.eval()
        val_loss, val_phys, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb, _, mb in tqdm(val_loader, desc=f"Val {epoch+1}", leave=False):
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                t = xb[:, -1, :]
                x_scaled, y_scaled = x_scaler.transform(xb), y_scaler.transform(yb)
                y_pred = model(x_scaled)
                loss_data = loss_fn(y_pred, y_scaled)
                y_pred_phys = y_scaler.inverse_transform(y_pred)
                a_pred = compute_second_derivative(y_pred_phys, t)
                a_grav = compute_gravity_acceleration(y_pred_phys, mb)[:, :, 1:-1]
                loss_phys = ((a_pred - a_grav) ** 2).mean()
                loss_phys_scaled = loss_phys * (loss_data.detach() / (loss_phys.detach() + 1e-8))
                loss = loss_data + phys_loss_weight * loss_phys_scaled
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

def test_ol_pinn(model, test_loader, x_scaler, y_scaler, device='cuda', log_path=None):
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb, _, _ in tqdm(test_loader, desc="Test"):
            xb, yb = xb.to(device), yb.to(device)
            x_scaled, y_scaled = x_scaler.transform(xb), y_scaler.transform(yb)
            y_pred = model(x_scaled)
            loss = loss_fn(y_pred, y_scaled)
            test_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    test_loss /= n
    print(f"[Test] Loss: {test_loss:.6f}")
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"test,{test_loss}\n")
    return test_loss

def run_ol_pinn_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dir = os.path.join(os.path.dirname(__file__), "data", "train")
    test_dir = os.path.join(os.path.dirname(__file__), "data", "test")
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

    dataset = TBPDataset(train_files)
    train_ds, val_ds, _ = random_split(dataset, [0.8, 0.2, 0], generator=torch.Generator().manual_seed(42))
    test_ds = TBPDataset(test_files)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    x_scaler, y_scaler = compute_scalers(train_loader, device)

    model = OL_PINN(
        deeponet_ckpt=os.path.join(os.path.dirname(__file__), 'models', 'deeponet.pth'),
        input_dim=22, hidden_dim=128, depth=4, output_dim=9, device=device
    )

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'ol_pinn.pth')
    log_path = os.path.join(os.path.dirname(__file__), 'logs', 'ol_pinn.log')

    train_ol_pinn(
        model, train_loader, val_loader, x_scaler, y_scaler,
        model_save_path=model_path,
        log_path=log_path,
        device=device
    )

    test_ol_pinn(model, test_loader, x_scaler, y_scaler, device=device, log_path=log_path)

if __name__ == '__main__':
    run_ol_pinn_pipeline()
