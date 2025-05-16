import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Tuple, Optional
from tqdm import tqdm
import random
import os

# ---------------------------------------
# 1) SpectralConv1d & FNO1d 정의
# ---------------------------------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, N)
        batch, _, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, in_channels, N//2+1)
        out_ft = torch.zeros(batch, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = torch.einsum(
            "bci,cio->boi", x_ft[:, :, :self.modes1], self.weights
        )
        x = torch.fft.irfft(out_ft, n=N, dim=-1)  # (batch, out_channels, N)
        return x

class FNO1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int, modes1: int, depth: int):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.spectral_blocks = nn.ModuleList([SpectralConv1d(width, width, modes1) for _ in range(depth)])
        self.pointwise_blocks = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(depth)])
        self.proj1 = nn.Conv1d(width, width // 2, kernel_size=1)
        self.proj2 = nn.Conv1d(width // 2, out_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for spec, pw in zip(self.spectral_blocks, self.pointwise_blocks):
            x1 = spec(x)
            x2 = pw(x)
            x = self.activation(x + x1 + x2)
        x = self.activation(self.proj1(x))
        x = self.proj2(x)
        return x

# ---------------------------------------
# 2) 데이터 로드 & 전처리
# ---------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_three_body_fno(
    csv_path: str, num_conditions: int = 500, time_steps: int = 2000, input_idx: slice = slice(18, 27), output_idx: slice = slice(0, 9)
) -> Tuple[np.ndarray, np.ndarray]:
    arr = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.float32)
    arr = arr.reshape(num_conditions, time_steps, -1)
    a_seq = arr[:, :, input_idx]  # (num_conditions, time_steps, in_channels)
    r_seq = arr[:, :, output_idx]  # (num_conditions, time_steps, out_channels)
    a_seq = np.transpose(a_seq, (0, 2, 1))  # (num_conditions, in_channels, time_steps)
    r_seq = np.transpose(r_seq, (0, 2, 1))
    return a_seq, r_seq

def prepare_fno_dataloaders(a_seq: np.ndarray, r_seq: np.ndarray, batch_size: int = 8):
    tensor_x = torch.from_numpy(a_seq)
    tensor_y = torch.from_numpy(r_seq)
    dataset = TensorDataset(tensor_x, tensor_y)
    N = len(dataset)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ---------------------------------------
# 3) 학습·검증·테스트 루프
# ---------------------------------------
def train_fno(
    csv_path: str = "data.csv",
    batch_size: int = 8,
    width: int = 64,
    modes1: int = 32,
    depth: int = 4,
    lr: float = 1e-3,
    num_epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
    num_conditions: int = 500,
    time_steps: int = 2000,
    input_idx: slice = slice(18, 27),
    output_idx: slice = slice(0, 9),
    model_save_path: str = "fno_threebody.pth",
    log_path: Optional[str] = "fno_train_log.csv"
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 데이터 준비
    a_seq, r_seq = load_three_body_fno(
        csv_path, num_conditions, time_steps, input_idx, output_idx
    )
    in_channels = a_seq.shape[1]
    out_channels = r_seq.shape[1]
    train_loader, val_loader, test_loader = prepare_fno_dataloaders(a_seq, r_seq, batch_size)

    # 모델 초기화
    model = FNO1d(in_channels, out_channels, width, modes1, depth).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 로그 초기화
    if log_path is not None:
        with open(log_path, "w") as f:
            f.write("epoch,train_mse,val_mse\n")

    best_val = float("inf")
    epochs_no_improve = 0
    for epoch in range(1, num_epochs+1):
        # 학습
        model.train()
        train_losses = []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item() * x.size(0))
        train_mse = np.sum(train_losses) / len(train_loader.dataset)

        # 검증
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                val_losses.append(loss.item() * x.size(0))
        val_mse = np.sum(val_losses) / len(val_loader.dataset)

        # 로그 저장
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_mse},{val_mse}\n")

        # Early Stopping
        if val_mse < best_val:
            best_val = val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}] Train MSE: {train_mse:.4e}, Val MSE: {val_mse:.4e}")

    # 테스트
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_losses = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test", leave=False):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            test_losses.append(loss.item() * x.size(0))
    test_mse = np.sum(test_losses) / len(test_loader.dataset)
    print(f">>> Test MSE: {test_mse:.4e}")
    print(f"Saved best FNO model → {model_save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes1", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_conditions", type=int, default=500)
    parser.add_argument("--time_steps", type=int, default=2000)
    parser.add_argument("--model_save_path", type=str, default="fno_threebody.pth")
    parser.add_argument("--log_path", type=str, default="fno_train_log.csv")
    args = parser.parse_args()

    train_fno(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        width=args.width,
        modes1=args.modes1,
        depth=args.depth,
        lr=args.lr,
        num_epochs=args.num_epochs,
        patience=args.patience,
        seed=args.seed,
        num_conditions=args.num_conditions,
        time_steps=args.time_steps,
        model_save_path=args.model_save_path,
        log_path=args.log_path,
    )
