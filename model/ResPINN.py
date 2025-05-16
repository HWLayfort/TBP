import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import random
import os

# Seed 고정 함수
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Residual Block 정의
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        return x + self.activation(self.layer(x))

# Res-PINN 모델 구현
class ResPINN(nn.Module):
    def __init__(self, hidden_dim=64, num_hidden_layers=4, input_dim=1, output_dim=9):
        super(ResPINN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()
        
    def forward(self, t):
        x = self.activation(self.input_layer(t))
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)

def compute_derivatives(model, t):
    t.requires_grad_(True)
    r = model(t)  # shape: (N, 9)
    v = torch.zeros_like(r)
    for i in range(r.shape[1]):
        grad_i = torch.autograd.grad(r[:, i].sum(), t, create_graph=True)[0]
        v[:, i] = grad_i.squeeze(-1)
    a = torch.zeros_like(r)
    for i in range(r.shape[1]):
        grad_i = torch.autograd.grad(v[:, i].sum(), t, create_graph=True)[0]
        a[:, i] = grad_i.squeeze(-1)
    return r, v, a

def physics_residual(r, a, masses, G=1.0):
    r = r.view(-1, 3, 3)
    a = a.view(-1, 3, 3)
    N, num_obj, _ = r.shape
    diff = r.unsqueeze(2) - r.unsqueeze(1)
    norm = torch.norm(diff, dim=-1, keepdim=True) + 1e-6
    masses_tensor = torch.tensor(masses, device=r.device).view(1, 1, num_obj, 1)
    a_grav = G * masses_tensor * diff / (norm ** 3)
    mask = 1 - torch.eye(num_obj, device=r.device).view(1, num_obj, num_obj, 1)
    a_grav = (a_grav * mask).sum(dim=2)
    return (a - a_grav).view(-1, 9)

def loss_function(data_r, data_v, data_a, pred_r, pred_v, pred_a, phys_res):
    mse = nn.MSELoss()
    loss_data = mse(pred_r, data_r) + mse(pred_v, data_v) + mse(pred_a, data_a)
    loss_phys = mse(phys_res, torch.zeros_like(phys_res))
    return loss_data + loss_phys

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    data = df.to_numpy(dtype=np.float32)
    data_r = data[:, 0:9]
    data_v = data[:, 9:18]
    data_a = data[:, 18:27]
    return data_r, data_v, data_a

def train_respinn(
    csv_path="data.csv",
    T=10.0,
    hidden_dim=64,
    num_hidden_layers=4,
    input_dim=1,
    output_dim=9,
    lr=1e-3,
    num_epochs=1000,
    batch_size=1024,
    seed=42,
    model_save_path="respinn_model.pth",
    log_path="respinn_train_log.csv",
    early_stopping_patience=100
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_r_np, data_v_np, data_a_np = load_data(csv_path)
    N = data_r_np.shape[0]
    t_values = np.linspace(0, T, N, dtype=np.float32).reshape(-1, 1)
    data_r = torch.from_numpy(data_r_np)
    data_v = torch.from_numpy(data_v_np)
    data_a = torch.from_numpy(data_a_np)
    t_data = torch.from_numpy(t_values)
    dataset = TensorDataset(t_data, data_r, data_v, data_a)
    train_size = int(0.8 * N)
    val_size = int(0.1 * N)
    test_size = N - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = ResPINN(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    masses = [1.0, 1.0, 1.0]

    best_val = float("inf")
    epochs_no_improve = 0

    # 로그 파일 초기화
    if log_path is not None:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            t_batch, data_r_batch, data_v_batch, data_a_batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            pred_r, pred_v, pred_a = compute_derivatives(model, t_batch)
            phys_res = physics_residual(pred_r, pred_a, masses)
            loss = loss_function(data_r_batch, data_v_batch, data_a_batch, pred_r, pred_v, pred_a, phys_res)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                t_batch, data_r_batch, data_v_batch, data_a_batch = [b.to(device) for b in batch]
                pred_r, pred_v, pred_a = compute_derivatives(model, t_batch)
                phys_res = physics_residual(pred_r, pred_a, masses)
                loss = loss_function(data_r_batch, data_v_batch, data_a_batch, pred_r, pred_v, pred_a, phys_res)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # 로그 기록
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss},{val_loss}\n")

        # Early Stopping
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
            if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    print(f"Model saved to {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", leave=False):
            t_batch, data_r_batch, data_v_batch, data_a_batch = [b.to(device) for b in batch]
            pred_r, pred_v, pred_a = compute_derivatives(model, t_batch)
            phys_res = physics_residual(pred_r, pred_a, masses)
            loss = loss_function(data_r_batch, data_v_batch, data_a_batch, pred_r, pred_v, pred_a, phys_res)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss = {test_loss:.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data.csv")
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_save_path", type=str, default="respinn_model.pth")
    parser.add_argument("--log_path", type=str, default="respinn_train_log.csv")
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    args = parser.parse_args()
    train_respinn(
        csv_path=args.csv_path,
        T=args.T,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        model_save_path=args.model_save_path,
        log_path=args.log_path,
        early_stopping_patience=args.early_stopping_patience
    )
