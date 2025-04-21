import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

#########################################
# Residual Block 정의
#########################################
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return self.activation(x + out)

#########################################
# Residual PINN (rPINN) 모델 구현
#########################################
class rPINN(nn.Module):
    def __init__(self, input_dim=1, output_dim=9, hidden_dim=64, num_res_blocks=4):
        nn.Module.__init__(self)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()
    
    def forward(self, t):
        # t: (batch_size, 1)
        x = self.activation(self.input_layer(t))
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)

#########################################
# 각 출력 성분별 미분 계산 (개별 미분 계산)
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
# 벡터화된 방식의 물리 잔차 (뉴턴의 만유인력 법칙)
#########################################
def physics_residual(r, a, masses, G=1.0):
    r = r.view(-1, 3, 3)  # (N, 3, 3)
    a = a.view(-1, 3, 3)  # (N, 3, 3)
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

#########################################
# CSV 파일로부터 데이터 로드 (r, v, a)
#########################################
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    data = df.to_numpy(dtype=np.float32)  # (N, 27)
    data_r = data[:, 0:9]
    data_v = data[:, 9:18]
    data_a = data[:, 18:27]
    return data_r, data_v, data_a

#########################################
# 전체 데이터를 학습/검증/테스트 셋으로 분할하고 DataLoader 구성
#########################################
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

#########################################
# 메인 함수: rPINN 학습, 모델 저장, 테스트 평가
#########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = "data.csv"  # data.csv는 python 파일과 동일한 경로에 존재
    T = 10.0  # 실제 관측 시간 범위에 맞게 수정 필요
    
    train_loader, val_loader, test_loader = prepare_dataloaders(csv_path, T)
    
    model = rPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    masses = [1.0, 1.0, 1.0]
    num_epochs = 1000
    
    for epoch in range(num_epochs):
        model.eval()
        test_loss = 0.0
        with torch.enable_grad():  # no_grad() 대신 enable_grad() 사용
            for batch in test_loader:
                t_batch, data_r_batch, data_v_batch, data_a_batch = [b.to(device) for b in batch]
                pred_r, pred_v, pred_a = compute_derivatives(model, t_batch)
                phys_res = physics_residual(pred_r, pred_a, masses)
                loss = loss_function(data_r_batch, data_v_batch, data_a_batch, pred_r, pred_v, pred_a, phys_res)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f"Test Loss = {test_loss:.6f}")
        
        # 검증 루프 (간단히 loss 출력)
        model.eval()
        val_loss = 0.0
        with torch.enable_grad():
            for batch in val_loader:
                t_batch, data_r_batch, data_v_batch, data_a_batch = [b.to(device) for b in batch]
                pred_r, pred_v, pred_a = compute_derivatives(model, t_batch)
                phys_res = physics_residual(pred_r, pred_a, masses)
                loss = loss_function(data_r_batch, data_v_batch, data_a_batch, pred_r, pred_v, pred_a, phys_res)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # 학습 완료 후 모델 저장
    torch.save(model.state_dict(), "rpinn_model.pth")
    print("Model saved to rpinn_model.pth")
    
    # 테스트 데이터 평가
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
