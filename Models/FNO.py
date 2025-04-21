import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

# ---------------------------------------
# 1) SpectralConv1d & FNO1d 정의
# ---------------------------------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        # 복소수 가중치
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, dtype=torch.cfloat))

    def forward(self, x):
        # x: (batch, in_channels, N)
        batch, _, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)                                # → (batch, in_channels, N//2+1)
        out_ft = torch.zeros(batch, self.out_channels, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = torch.einsum(
            "bci,cio->boi", x_ft[:, :, :self.modes1], self.weights
        )
        x = torch.fft.irfft(out_ft, n=N, dim=-1)                       # → (batch, out_channels, N)
        return x

class FNO1d(nn.Module):
    def __init__(self, in_channels, out_channels, width, modes1, depth):
        nn.Module.__init__(self)
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.spectral_blocks = nn.ModuleList()
        self.pointwise_blocks = nn.ModuleList()
        for _ in range(depth):
            self.spectral_blocks.append(SpectralConv1d(width, width, modes1))
            self.pointwise_blocks.append(nn.Conv1d(width, width, kernel_size=1))
        self.proj1 = nn.Conv1d(width, width // 2, kernel_size=1)
        self.proj2 = nn.Conv1d(width // 2, out_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (batch, in_channels, N)
        x = self.lift(x)                                               # → (batch, width, N)
        for spec, pw in zip(self.spectral_blocks, self.pointwise_blocks):
            x1 = spec(x)
            x2 = pw(x)
            x = self.activation(x + x1 + x2)
        x = self.activation(self.proj1(x))
        x = self.proj2(x)                                              # → (batch, out_channels, N)
        return x

# ---------------------------------------
# 2) 데이터 로드 & 전처리
# ---------------------------------------
def load_three_body_fno(csv_path, num_conditions=500, time_steps=2000):
    # CSV에 header 없다고 가정
    arr = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.float32)
    arr = arr.reshape(num_conditions, time_steps, 27)               # (500,2000,27)
    # 입력: a1,a2,a3 (마지막 9열)
    a_seq = arr[:, :, 18:27]                                        # (500,2000,9)
    # 출력: r1,r2,r3 (첫 9열)
    r_seq = arr[:, :, 0:9]                                          # (500,2000,9)
    # 차원 순서 변경 → (500, 9, 2000)
    a_seq = np.transpose(a_seq, (0, 2, 1))
    r_seq = np.transpose(r_seq, (0, 2, 1))
    return a_seq, r_seq

# ---------------------------------------
# 3) DataLoader 준비 (80/10/10 split)
# ---------------------------------------
def prepare_fno_dataloaders(a_seq, r_seq, batch_size=8):
    tensor_x = torch.from_numpy(a_seq)                             # (500,9,2000)
    tensor_y = torch.from_numpy(r_seq)                             # (500,9,2000)
    dataset = TensorDataset(tensor_x, tensor_y)
    N = len(dataset)
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)
    n_test  = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ---------------------------------------
# 4) 학습·검증·테스트 루프
# ---------------------------------------
def train_fno(csv_path="data.csv"):
    # 1) 데이터 로드
    a_seq, r_seq = load_three_body_fno(csv_path)
    train_loader, val_loader, test_loader = prepare_fno_dataloaders(a_seq, r_seq, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNO1d(in_channels=9, out_channels=9, width=64, modes1=32, depth=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    num_epochs = 200

    for epoch in range(1, num_epochs+1):
        # — train
        model.train()
        total_train = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * x.size(0)
        train_mse = total_train / len(train_loader.dataset)

        # — val
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                total_val += loss_fn(model(x), y).item() * x.size(0)
        val_mse = total_val / len(val_loader.dataset)

        # best 모델 저장
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), "fno_threebody.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}] Train MSE: {train_mse:.4e}, Val MSE: {val_mse:.4e}")

    # — test
    model.load_state_dict(torch.load("fno_threebody.pth"))
    model.eval()
    total_test = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            total_test += loss_fn(model(x), y).item() * x.size(0)
    test_mse = total_test / len(test_loader.dataset)
    print(f">>> Test MSE: {test_mse:.4e}")
    print("Saved best FNO model → fno_threebody.pth")

if __name__ == "__main__":
    train_fno("data.csv")
