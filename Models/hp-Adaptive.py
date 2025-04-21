import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

#########################################
# 기본 PINN 모델 (측정 데이터용)
#########################################
class PINN(nn.Module):
    def __init__(self, hidden_dim=64, num_hidden_layers=4, input_dim=1, output_dim=9):
        # p-adaptivity: 필요시 hidden_dim 또는 num_hidden_layers를 증가시켜 네트워크 용량을 확장할 수 있음
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

#########################################
# 각 성분에 대해 미분 계산 (각각의 출력에 대해 개별 미분)
#########################################
def compute_derivatives(model, t):
    t.requires_grad_(True)
    r = model(t)  # shape: (batch, 9)
    v = torch.zeros_like(r)
    for i in range(r.shape[1]):
        grad_i = torch.autograd.grad(r[:, i].sum(), t, create_graph=True)[0]
        v[:, i] = grad_i.squeeze(-1)
    a = torch.zeros_like(r)
    for i in range(r.shape[1]):
        grad_i = torch.autograd.grad(v[:, i].sum(), t, create_graph=True)[0]
        a[:, i] = grad_i.squeeze(-1)
    return r, v, a

#########################################
# 벡터화된 물리 잔차 계산 (뉴턴의 만유인력 법칙)
#########################################
def physics_residual(r, a, masses, G=1.0):
    r = r.view(-1, 3, 3)  # (N, 3, 3)
    a = a.view(-1, 3, 3)
    N, num_obj, _ = r.shape
    diff = r.unsqueeze(2) - r.unsqueeze(1)  # (N, 3, 3, 3)
    norm = torch.norm(diff, dim=-1, keepdim=True) + 1e-6  # (N, 3, 3, 1)
    masses_tensor = torch.tensor(masses, device=r.device).view(1, 1, num_obj, 1)
    a_grav = G * masses_tensor * diff / (norm ** 3)  # (N, 3, 3, 3)
    mask = 1 - torch.eye(num_obj, device=r.device).view(1, num_obj, num_obj, 1)
    a_grav = (a_grav * mask).sum(dim=2)  # (N, 3, 3)
    return (a - a_grav).view(-1, 9)

#########################################
# 손실 함수: 측정 데이터 손실 + 물리 손실
#########################################
def loss_function(data_r, data_v, data_a, pred_r, pred_v, pred_a, phys_res):
    mse = nn.MSELoss()
    loss_data = mse(pred_r, data_r) + mse(pred_v, data_v) + mse(pred_a, data_a)
    loss_phys = mse(phys_res, torch.zeros_like(phys_res))
    return loss_data + loss_phys

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    data = df.to_numpy(dtype=np.float32)  # (N, 27)
    data_r = data[:, 0:9]
    data_v = data[:, 9:18]
    data_a = data[:, 18:27]
    return data_r, data_v, data_a

def adaptive_collocation_sampling(model, device, masses, T, num_dense=10000, threshold_percentile=90):
    model.eval()
    with torch.no_grad():
        dense_t = torch.linspace(0, T, num_dense, dtype=torch.float32).view(-1, 1).to(device)
        r_dense, _, a_dense = compute_derivatives(model, dense_t)
        res_dense = physics_residual(r_dense, a_dense, masses)  # (num_dense, 9)
        error = torch.mean(torch.abs(res_dense), dim=1)  # (num_dense,)
        threshold = torch.quantile(error, threshold_percentile/100.0)
        new_points = dense_t[error > threshold]
    model.train()
    return new_points.cpu()

def adaptive_training(model, optimizer, masses, T, meas_loader, device, 
                      init_collocation=None, adapt_interval=200, max_adapt_iters=5):
    if init_collocation is None:
        init_collocation = torch.linspace(0, T, 10000, dtype=torch.float32).view(-1, 1)
    collocation_t = init_collocation.clone()
    
    colloc_dataset = TensorDataset(collocation_t)
    colloc_loader = DataLoader(colloc_dataset, batch_size=1024, shuffle=True)
    
    num_epochs_total = 1000
    adapt_iters = 0
    for epoch in range(num_epochs_total):
        model.train()
        train_loss = 0.0
        for batch in meas_loader:
            t_meas, data_r_meas, data_v_meas, data_a_meas = [b.to(device) for b in batch]
            optimizer.zero_grad()
            pred_r, pred_v, pred_a = compute_derivatives(model, t_meas)
            try:
                colloc_batch = next(colloc_iter)
            except:
                colloc_iter = iter(colloc_loader)
                colloc_batch = next(colloc_iter)
            t_colloc = colloc_batch[0].to(device)
            pred_r_colloc, _, pred_a_colloc = compute_derivatives(model, t_colloc)
            phys_res = physics_residual(pred_r_colloc, pred_a_colloc, masses)
            
            loss = loss_function(data_r_meas, data_v_meas, data_a_meas, 
                                 pred_r, pred_v, pred_a, phys_res)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(meas_loader)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")
        
        if (epoch > 0) and (epoch % adapt_interval == 0) and (adapt_iters < max_adapt_iters):
            new_colloc = adaptive_collocation_sampling(model, device, masses, T, num_dense=10000, threshold_percentile=90)
            print(f"Adaptive iteration {adapt_iters+1}: adding {new_colloc.shape[0]} new collocation points.")
            collocation_t = torch.cat([collocation_t, new_colloc.cpu()], dim=0)
            colloc_dataset = TensorDataset(collocation_t)
            colloc_loader = DataLoader(colloc_dataset, batch_size=1024, shuffle=True)
            adapt_iters += 1
            
            if adapt_iters == max_adapt_iters:
                print("p-adaptivity: Increasing network capacity.")
                new_model = PINN(hidden_dim=128, num_hidden_layers=6).to(device)
                new_model.load_state_dict(model.state_dict(), strict=False)
                model = new_model
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model

def prepare_dataloaders(csv_path, T):
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
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = "data.csv"
    T = 10.0
    train_loader, val_loader, test_loader = prepare_dataloaders(csv_path, T)
    
    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    masses = [1.0, 1.0, 1.0]
    
    model = adaptive_training(model, optimizer, masses, T, train_loader, device,
                              init_collocation=None, adapt_interval=200, max_adapt_iters=5)
    
    # 모델 저장
    torch.save(model.state_dict(), "pinn_model_hp_adaptive.pth")
    print("Model saved to pinn_model_hp_adaptive.pth")
    
    # 테스트 평가
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
