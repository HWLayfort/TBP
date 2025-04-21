import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

# PINN 모델 구현 (nn.Module.__init__(self) 직접 호출)
class PINN(nn.Module):
    def __init__(self, hidden_dim=64, num_hidden_layers=4, input_dim=1, output_dim=9):
        nn.Module.__init__(self)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()
        
    def forward(self, t):
        x = self.activation(self.input_layer(t))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    csv_path = "data.csv"
    data_r_np, data_v_np, data_a_np = load_data(csv_path)
    N = data_r_np.shape[0]
    T = 10.0
    t_values = np.linspace(0, T, N, dtype=np.float32).reshape(-1, 1)
    
    data_r = torch.from_numpy(data_r_np)
    data_v = torch.from_numpy(data_v_np)
    data_a = torch.from_numpy(data_a_np)
    t_data = torch.from_numpy(t_values)
    
    dataset = TensorDataset(t_data, data_r, data_v, data_a)
    train_size = int(0.8 * N)
    val_size = int(0.1 * N)
    test_size = N - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    masses = [1.0, 1.0, 1.0]
    num_epochs = 1000
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
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
            for batch in val_loader:
                t_batch, data_r_batch, data_v_batch, data_a_batch = [b.to(device) for b in batch]
                pred_r, pred_v, pred_a = compute_derivatives(model, t_batch)
                phys_res = physics_residual(pred_r, pred_a, masses)
                loss = loss_function(data_r_batch, data_v_batch, data_a_batch, pred_r, pred_v, pred_a, phys_res)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    torch.save(model.state_dict(), "pinn_model.pth")
    print("Model saved to pinn_model.pth")
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            t_batch, data_r_batch, data_v_batch, data_a_batch = [b.to(device) for b in batch]
            pred_r, pred_v, pred_a = compute_derivatives(model, t_batch)
            phys_res = physics_residual(pred_r, pred_a, masses)
            loss = loss_function(data_r_batch, data_v_batch, data_a_batch, pred_r, pred_v, pred_a, phys_res)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss = {test_loss:.6f}")

if __name__ == "__main__":
    main()
