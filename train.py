# train.py
import argparse
import torch, torch.optim as optim, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np, pandas as pd

from models import (
    DeepONet, FNO1d, PINN, rPINN,
    compute_derivatives, physics_residual
)

# ---------------------------------------
# 1) DeepONet 데이터 로더
# ---------------------------------------
def load_deeponet(csv_path, nc=500, ts=2000):
    arr = pd.read_csv(csv_path, header=None).to_numpy(np.float32)
    arr = arr.reshape(nc, ts, 27)
    init = arr[:, 0, :]
    T = 10.0
    t_lin = np.linspace(0, T, ts, dtype=np.float32).reshape(-1,1)
    B = nc * ts
    branch = np.repeat(init, ts, axis=0)
    trunk  = np.tile(t_lin, (nc,1))
    target = arr[:,:,0:9].reshape(B, 9)
    return branch, trunk, target

# ---------------------------------------
# 2) FNO 데이터 로더
# ---------------------------------------
def load_fno(csv_path, nc=500, ts=2000):
    arr = pd.read_csv(csv_path, header=None).to_numpy(np.float32)
    arr = arr.reshape(nc, ts, 27)
    a = np.transpose(arr[:,:,18:27], (0,2,1))
    r = np.transpose(arr[:,:, 0:9], (0,2,1))
    return a, r

# ---------------------------------------
# 3) 일반 모델용 DataLoader 생성
# ---------------------------------------
def make_loader(*arrays, batch_size=1024, shuffle=True):
    tensors = [torch.from_numpy(x) for x in arrays]
    ds = TensorDataset(*tensors)
    n = len(ds)
    n_tr = int(0.8 * n); n_va = int(0.1 * n)
    n_te = n - n_tr - n_va
    tr, va, te = random_split(ds, [n_tr,n_va,n_te])
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=shuffle),
        DataLoader(va, batch_size=batch_size, shuffle=False),
        DataLoader(te, batch_size=batch_size, shuffle=False),
    )

# ---------------------------------------
# 4) 일반 모델 학습 루프
# ---------------------------------------
def train_loop(model, loaders, loss_fn, optimizer, epochs, device):
    tr, va, te = loaders
    best = float('inf')
    for e in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for batch in tr:
            optimizer.zero_grad()
            out = model(*[b.to(device) for b in batch[:-1]])
            loss = loss_fn(out, batch[-1].to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch[0].size(0)
        train_loss /= len(tr.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in va:
                out = model(*[b.to(device) for b in batch[:-1]])
                val_loss += loss_fn(out, batch[-1].to(device)).item() * batch[0].size(0)
        val_loss /= len(va.dataset)

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), f"{args.model}_best.pth")
        if e==1 or e%10==0:
            print(f"[{e}/{epochs}] Train {train_loss:.3e}  Val {val_loss:.3e}")

    # 테스트 평가
    model.load_state_dict(torch.load(f"{args.model}_best.pth"))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in te:
            out = model(*[b.to(device) for b in batch[:-1]])
            test_loss += loss_fn(out, batch[-1].to(device)).item() * batch[0].size(0)
    print("Test MSE:", test_loss / len(te.dataset))

# ---------------------------------------
# 5) hp-Adaptive 전용 함수들
# ---------------------------------------
def loss_function(data_r, data_v, data_a, pred_r, pred_v, pred_a, phys_res):
    mse = nn.MSELoss()
    loss_data = mse(pred_r, data_r) + mse(pred_v, data_v) + mse(pred_a, data_a)
    loss_phys = mse(phys_res, torch.zeros_like(phys_res))
    return loss_data + loss_phys

def load_data(csv_path):
    df = pd.read_csv(csv_path, header=None).to_numpy(np.float32)
    data_r = df[:, 0:9]
    data_v = df[:, 9:18]
    data_a = df[:, 18:27]
    return data_r, data_v, data_a

def prepare_dataloaders(csv_path, T=10.0):
    data_r_np, data_v_np, data_a_np = load_data(csv_path)
    N = data_r_np.shape[0]
    t_vals = np.linspace(0, T, N, dtype=np.float32).reshape(-1,1)

    t = torch.from_numpy(t_vals)
    r = torch.from_numpy(data_r_np)
    v = torch.from_numpy(data_v_np)
    a = torch.from_numpy(data_a_np)

    ds = TensorDataset(t, r, v, a)
    n_tr = int(0.8*N); n_va = int(0.1*N); n_te = N - n_tr - n_va
    tr, va, te = random_split(ds, [n_tr,n_va,n_te])

    return (
        DataLoader(tr, batch_size=1024, shuffle=True),
        DataLoader(va, batch_size=1024, shuffle=False),
        DataLoader(te, batch_size=1024, shuffle=False),
    )

def adaptive_collocation_sampling(model, device, masses, T,
                                  num_dense=10000, threshold_percentile=90):
    model.eval()
    with torch.no_grad():
        dense_t = torch.linspace(0, T, num_dense, dtype=torch.float32).view(-1, 1).to(device)
        r_d, _, a_d = compute_derivatives(model, dense_t)
        res = physics_residual(r_d, a_d, masses)
        err = torch.mean(torch.abs(res), dim=1)
        th = torch.quantile(err, threshold_percentile/100.0)
        new_t = dense_t[err > th]
    model.train()
    return new_t.cpu()

def adaptive_training(model, optimizer, masses, T, meas_loader, device,
                      init_collocation=None, adapt_interval=200, max_adapt_iters=5):
    if init_collocation is None:
        init_collocation = torch.linspace(0, T, 10000, dtype=torch.float32).view(-1, 1)
    colloc_t = init_collocation.clone()
    colloc_loader = DataLoader(TensorDataset(colloc_t), batch_size=1024, shuffle=True)
    colloc_iter = iter(colloc_loader)
    adapt_it = 0
    total_epochs = 1000

    for epoch in range(total_epochs):
        model.train()
        train_loss = 0.0
        for t_meas, d_r, d_v, d_a in meas_loader:
            t_meas, d_r, d_v, d_a = [x.to(device) for x in (t_meas, d_r, d_v, d_a)]
            optimizer.zero_grad()
            pr, pv, pa = compute_derivatives(model, t_meas)

            try:
                t_colloc = next(colloc_iter)[0].to(device)
            except StopIteration:
                colloc_iter = iter(colloc_loader)
                t_colloc = next(colloc_iter)[0].to(device)

            prc, _, pac = compute_derivatives(model, t_colloc)
            phys_res = physics_residual(prc, pac, masses)

            loss = loss_function(d_r, d_v, d_a, pr, pv, pa, phys_res)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss/len(meas_loader):.6f}")

        if epoch>0 and epoch % adapt_interval==0 and adapt_it < max_adapt_iters:
            new_t = adaptive_collocation_sampling(model, device, masses, T)
            print(f"Adaptive iteration {adapt_it+1}: added {new_t.shape[0]} points.")
            colloc_t = torch.cat([colloc_t, new_t.cpu()], dim=0)
            colloc_loader = DataLoader(TensorDataset(colloc_t), batch_size=1024, shuffle=True)
            colloc_iter = iter(colloc_loader)
            adapt_it += 1

            if adapt_it == max_adapt_iters:
                print("p-adaptivity: increasing network capacity.")
                new_model = PINN(hidden_dim=128, num_hidden_layers=6).to(device)
                new_model.load_state_dict(model.state_dict(), strict=False)
                model = new_model
                optimizer = optim.Adam(model.parameters(), lr=1e-3)

    return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["deeponet","fno","pinn","rpinn"], required=True)
    p.add_argument("--csv",    default="data.csv")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch",  type=int, default=1024)
    p.add_argument("--lr",     type=float, default=1e-3)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 10.0
    masses = [1.0, 1.0, 1.0]

    if args.model == "deeponet":
        br, tr, tar = load_deeponet(args.csv)
        loaders = make_loader(br, tr, tar, batch_size=args.batch)
        model = DeepONet(branch_input_dim=27, trunk_input_dim=1).to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_loop(model, loaders, loss_fn, optimizer, args.epochs, device)

    elif args.model == "fno":
        a, r = load_fno(args.csv)
        loaders = make_loader(a, r, batch_size=args.batch)
        model = FNO1d(in_channels=9, out_channels=9, width=64, modes1=32, depth=4).to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_loop(model, loaders, loss_fn, optimizer, args.epochs, device)

    else:  # pinn / rpinn
        train_loader, val_loader, test_loader = prepare_dataloaders(args.csv, T)
        model = (PINN().to(device) if args.model=="pinn" else rPINN().to(device))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = adaptive_training(model, optimizer, masses, T, train_loader, device)

        # 최종 테스트
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for t_b, dr_b, dv_b, da_b in test_loader:
                t_b, dr_b, dv_b, da_b = [x.to(device) for x in (t_b, dr_b, dv_b, da_b)]
                pr, pv, pa = compute_derivatives(model, t_b)
                phys = physics_residual(pr, pa, masses)
                test_loss += loss_function(dr_b, dv_b, da_b, pr, pv, pa, phys).item()
        print(f"Final Test Loss = {test_loss/len(test_loader):.6f}")
