import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ======== 데이터셋 정의 ==========
class ThreeBodyDataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        filename = os.path.basename(csv_path)
        base = filename.split('_')[0]
        mass_str = base.replace('m', '')
        self.masses = np.array([float(x) for x in mass_str.split('-')])
        df = pd.read_csv(csv_path, header=None)
        self.t = df.iloc[:, 0].values
        self.r = df.iloc[:, 1:10].values.reshape(-1, 3, 3)
        self.v = df.iloc[:, 10:19].values.reshape(-1, 3, 3)
        self.r0 = self.r[0]
        self.v0 = self.v[0]
        self.length = len(df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {
            't': torch.tensor(self.t[idx], dtype=torch.float32),       # ()
            'r': torch.tensor(self.r[idx], dtype=torch.float32),       # (3, 3)
            'v': torch.tensor(self.v[idx], dtype=torch.float32),       # (3, 3)
            'm': torch.tensor(self.masses, dtype=torch.float32),       # (3,)
            'r0': torch.tensor(self.r0, dtype=torch.float32),          # (3, 3)
            'v0': torch.tensor(self.v0, dtype=torch.float32)           # (3, 3)
        }
        return sample

class ThreeBodyPINNDataset(Dataset):
    def __init__(self, filelist):
        self.datasets = [ThreeBodyDataset(f) for f in filelist]
        self.all_data = []
        for ds in self.datasets:
            for i in range(len(ds)):
                d = ds[i]
                input_vec = np.concatenate([
                    np.array([d['t']]),
                    d['r0'].reshape(-1),
                    d['v0'].reshape(-1),
                    d['m'].reshape(-1)
                ])
                output_vec = d['r'].reshape(-1)
                # v는 optional로 필요하다면 추가 가능
                v_vec = d['v'].reshape(-1)
                self.all_data.append((input_vec, output_vec, v_vec, d['m']))
    def __len__(self):
        return len(self.all_data)
    def __getitem__(self, idx):
        x, r, v, m = self.all_data[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(v, dtype=torch.float32),
            torch.tensor(m, dtype=torch.float32)
        )

# ======= PINN 모델 ==========
class PINN(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=128, num_hidden_layers=8, output_dim=9):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

# ======= 삼체 문제의 중력 가속도(Physics residual) ==========
def compute_physics_residual(r_pred, a_pred, masses, G=1.0, eps=1e-8):
    """
    r_pred: (batch, 9)  (3 bodies, 3D)
    a_pred: (batch, 9)  (3 bodies, 3D)
    masses: (batch, 3)
    """
    B = r_pred.shape[0]
    r = r_pred.view(B, 3, 3)
    a = a_pred.view(B, 3, 3)
    # 중력 가속도 계산
    m = masses.view(B, 1, 3, 1)  # (B,1,3,1)
    diff = r.unsqueeze(2) - r.unsqueeze(1)    # (B,3,3,3)
    norm = torch.norm(diff, dim=-1, keepdim=True) + eps   # (B,3,3,1)
    a_grav = G * m * diff / (norm ** 3)      # (B,3,3,3)
    mask = 1 - torch.eye(3, device=r.device).view(1, 3, 3, 1)
    a_grav = (a_grav * mask).sum(dim=2)      # (B,3,3)
    # 차이 반환: (a - a_grav), flatten
    return (a - a_grav).view(B, -1)

# ======= 자동 미분으로 a_pred 계산 ==========
def compute_acceleration(model, x, eps=1e-3):
    """
    입력 x에 대해 모델이 예측한 r(t)로부터 미분(2계)해서 a_pred 계산
    x: (batch, input_dim)
    """
    def shift_t(x, delta):
        x_new = x.clone()
        x_new[:, 0] = x[:, 0] + delta
        return x_new

    r      = model(x)
    r_p1   = model(shift_t(x, +eps))
    r_m1   = model(shift_t(x, -eps))
    r_p2   = model(shift_t(x, +2*eps))
    r_m2   = model(shift_t(x, -2*eps))

    a = (-r_m2 + 16*r_m1 - 30*r + 16*r_p1 - r_p2) / (12 * eps**2)
    return a

# ======= 평가 함수 ==========
def eval_loop(model, dataloader, device, loss_fn, weight_phys=1.0, desc="Eval", require_physics=False):
    model.eval()
    total_loss = 0.0
    total_phy_loss = 0.0
    n = 0
    for xb, yb, vb, mb in tqdm(dataloader, desc=desc):
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        # 예측 위치
        y_pred = model(xb)
        loss = loss_fn(y_pred, yb)
        # Physics residual loss
        if require_physics:
            xb_req = xb.clone().detach().requires_grad_(True)
            r_pred = model(xb_req)
            a_pred = compute_acceleration(model, xb_req)
            phy_res = compute_physics_residual(r_pred, a_pred, mb)
            phy_loss = (phy_res**2).mean()
            total_phy_loss += phy_loss.item() * xb.size(0)
            loss = loss + weight_phys * phy_loss
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    avg_loss = total_loss / n
    avg_phy_loss = total_phy_loss / n if require_physics else None
    return avg_loss, avg_phy_loss

def test_loop(model, dataloader, device, loss_fn, desc="Test"):
    """
    Test 단계에서는 오직 예측값과 실제값(ground-truth) 사이의 MSE만 평가한다.
    Physics residual(physics loss)는 포함하지 않는다.
    """
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb, vb, mb in tqdm(dataloader, desc=desc):
            xb = xb.to(device)
            yb = yb.to(device)
            # mb = mb.to(device) # masses, 사용하지 않음
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    avg_loss = total_loss / n
    return avg_loss

# ======= 학습 함수 ==========
def train_loop(
        model, 
        train_loader, 
        val_loader, 
        test_loader, 
        num_epochs=10, 
        lr=1e-3, 
        device='cuda', 
        weight_phys=1.0, 
        model_save_path='pinn.pth', 
        log_path='pinn.log',
        early_stop_patience=20
    ):
    print(f"Training PINN on device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    best_epoch = 0
    patience = 0
    
    if log_path is not None:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
    
    for epoch in range(num_epochs):
        # ----- Train -----
        model.train()
        total_loss = 0.0
        total_phy_loss = 0.0
        n = 0
        for xb, yb, vb, mb in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            # Physics residual
            xb_req = xb.clone().detach().requires_grad_(True)
            r_pred = model(xb_req)
            a_pred = compute_acceleration(model, xb_req)
            phy_res = compute_physics_residual(r_pred, a_pred, mb)
            phy_loss = (phy_res**2).mean()
            total_phy_loss += phy_loss.item() * xb.size(0)
            loss = loss + weight_phys * phy_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        avg_loss = total_loss / n
        avg_phy_loss = total_phy_loss / n
        
        val_loss, val_phy_loss = eval_loop(model, val_loader, device, loss_fn, weight_phys=weight_phys, desc="Val", require_physics=True)
        
        if epoch % 10 == 0: 
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.6f}, Train Physics Loss: {avg_phy_loss:.6f}, Val Loss: {val_loss:.6f}, Val Physics Loss: {val_phy_loss:.6f}")
            
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(f"{epoch},{avg_loss},{avg_phy_loss},{val_loss},{val_phy_loss}\n")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(model_save_path), model_save_path))
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}: no improvement in {early_stop_patience} epochs.")
                break
            
    # ----- Test -----
    test_loss = test_loop(model, test_loader, device, nn.MSELoss(), desc="Test (MSE only)")
    print(f"[Test] Final Test Loss (MSE only): {test_loss:.6f}")

    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"test,{test_loss}\n")

# ======= 전체 파이프라인 ==========
def run_pinn_pipeline():
    data_dir = os.path.join(os.path.dirname(__file__), "data", "output")
    file_list = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.csv')]
    dataset = ThreeBodyPINNDataset(file_list)
    print(f"Loaded dataset with {len(dataset)} samples.")
    # 데이터셋 분할
    total = len(dataset)
    n_train = int(0.7 * total)
    n_val = int(0.15 * total)
    n_test = total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = PINN(input_dim=22, hidden_dim=128, num_hidden_layers=8, output_dim=9)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loop(model, train_loader, val_loader, test_loader, num_epochs=500, lr=1e-3, device=device, weight_phys=1.0)

if __name__ == "__main__":
    run_pinn_pipeline()