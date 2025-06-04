from DataLoader import TBPDataset, compute_scalers
from FNO import FNO
from TFNO import TFNO
from DeepONet import DeepONet
from ResPINN import ResPINN
from PINN import compute_second_derivative, compute_gravity_acceleration

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

def train_multitask_fpinn(
    model, teacher, train_loader, val_loader,
    num_epochs=100, lr=1e-3, device='cuda',
    x_scaler=None, y_scaler=None,
    model_save_path='multi_pinn.pth', log_path='multi_pinn.log',
    early_stop_patience=30,
    lambda_data=1.0,
    lambda_phys=1e-1,
    teacher_model="fno"
):
    model = model.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    patience = 0
    with open(log_path, "w") as f:
        f.write("epoch,data_loss,phys_loss,val_data,val_phys\n")

    for epoch in range(num_epochs):
        model.train()
        total_data, total_phys, n = 0.0, 0.0, 0

        for xb, yb, _, mb in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            t = xb[:, -1, :]
            tb = xb[:, 21:, :].permute(0, 2, 1)
            x_scaled = x_scaler.transform(xb)
            x_cond = x_scaled[:, :21, 0]                 # (B, 21)
            t_cond = x_scaled[:, 21:, :].permute(0, 2, 1)     # (B, N, 1)

            with torch.no_grad():
                if teacher_model == "deeponet":
                    y_teacher = teacher(x_scaled[:, :21, :].mean(dim=2), tb)
                else:
                    y_teacher = teacher(x_scaled)

            optimizer.zero_grad()
            y_pred = model(x_cond, t_cond)
            loss_data = loss_fn(y_pred, y_teacher)  # PINO 예측값을 지도신호로 사용

            y_pred_phys = y_scaler.inverse_transform(y_pred)
            a_pred = compute_second_derivative(y_pred_phys, t)
            a_grav = compute_gravity_acceleration(y_pred_phys, mb)[:, :, 1:-1]
            loss_phys = ((a_pred - a_grav) ** 2).mean()
            loss_phys_scaled = loss_phys * (loss_data.detach() / (loss_phys.detach() + 1e-8))

            loss = lambda_data * loss_data + lambda_phys * loss_phys_scaled
            loss.backward()
            optimizer.step()

            total_data += loss_data.item() * xb.size(0)
            total_phys += loss_phys.item() * xb.size(0)
            n += xb.size(0)

        train_data = total_data / n
        train_phys = total_phys / n

        # Validation
        model.eval()
        val_data, val_phys, n = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb, _, mb in tqdm(val_loader, desc="Test", leave=False):
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                t = xb[:, -1, :]
                x_scaled, y_scaled = x_scaler.transform(xb), y_scaler.transform(yb)
                x_cond = x_scaled[:, :21, 0]                 # (B, 21)
                t_cond = x_scaled[:, 21:, :].permute(0, 2, 1)     # (B, N, 1)
                y_pred = model(x_cond, t_cond)
                loss_data = loss_fn(y_pred, y_scaled)

                y_pred_phys = y_scaler.inverse_transform(y_pred)
                a_pred = compute_second_derivative(y_pred_phys, t)
                a_grav = compute_gravity_acceleration(y_pred_phys, mb)[:, :, 1:-1]
                loss_phys = ((a_pred - a_grav) ** 2).mean()

                val_data += loss_data.item() * xb.size(0)
                val_phys += loss_phys.item() * xb.size(0)
                n += xb.size(0)

        val_data /= n
        val_phys /= n
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_data},{train_phys},{val_data},{val_phys}\n")

        if val_data < best_val:
            best_val = val_data
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if epoch % 10 == 0:
            print(f"[{epoch+1}] Train: Data={train_data:.6f}, Phys={train_phys:.6f} | Val: Data={val_data:.6f}, Phys={val_phys:.6f}")
    
    # load the best model
    model.load_state_dict(torch.load(model_save_path))

def test_multitask_fpinn(
    model, teacher, test_loader, x_scaler, y_scaler, device='cuda',
    log_path='multi_pinn.log'
):
    model.eval()
    teacher = teacher.to(device)
    loss_fn = nn.MSELoss()
    test_data, test_phys, n = 0.0, 0.0, 0

    with torch.no_grad():
        for xb, yb, _, mb in tqdm(test_loader, desc="Test"):
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            t = xb[:, -1, :]
            x_scaled, y_scaled = x_scaler.transform(xb), y_scaler.transform(yb)
            x_cond = x_scaled[:, :21, 0]                 # (B, 21)
            t_cond = x_scaled[:, 21:, :].permute(0, 2, 1)     # (B, N, 1)
            y_pred = model(x_cond, t_cond)
            loss_data = loss_fn(y_pred, y_scaled)

            y_pred_phys = y_scaler.inverse_transform(y_pred)
            a_pred = compute_second_derivative(y_pred_phys, t)
            a_grav = compute_gravity_acceleration(y_pred_phys, mb)[:, :, 1:-1]
            loss_phys = ((a_pred - a_grav) ** 2).mean()

            test_data += loss_data.item() * xb.size(0)
            test_phys += loss_phys.item() * xb.size(0)
            n += xb.size(0)

    test_data /= n
    test_phys /= n
    print(f"[Test] Data Loss: {test_data:.6f}, Phys Loss: {test_phys:.6f}")
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"test,{test_data},{test_phys}\n")
    return test_data, test_phys

def run_multitask_deeponet_pipeline(train_loader, val_loader, test_loader, x_scaler, y_scaler, device='cuda'):
    model = ResPINN(input_dim=21, embed_dim=32, hidden_dim=128, depth=6, output_dim=9)
    fno_teacher = FNO(modes=32, width=64, input_dim=22, output_dim=9, depth=4)
    fno_teacher.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'models', 'fno.pth')))
    fno_teacher.eval()

    train_multitask_fpinn(
        model, fno_teacher, train_loader, val_loader,
        num_epochs=500, lr=1e-3, device=device,
        x_scaler=x_scaler, y_scaler=y_scaler,
        model_save_path=os.path.join(os.path.dirname(__file__), 'models', 'fno_to_respinn.pth'),
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'fno_to_respinn.log'),
        lambda_data=1.0,
        lambda_phys=1e-5
    )

    test_multitask_fpinn(
        model, fno_teacher, test_loader,
        x_scaler=x_scaler, y_scaler=y_scaler,
        device=device,
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'fno_to_respinn.log')
    )
    
    model = ResPINN(input_dim=21, embed_dim=32, hidden_dim=128, depth=6, output_dim=9)
    tfno_teacher = TFNO(
        t_steps=100, input_dim=22, embed_dim=32, width=64, modes=32,
        depth=4, output_dim=9
    )
    tfno_teacher.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'models', 'tfno.pth')))
    tfno_teacher.eval()
    
    train_multitask_fpinn(
        model, tfno_teacher, train_loader, val_loader,
        num_epochs=500, lr=1e-3, device=device,
        x_scaler=x_scaler, y_scaler=y_scaler,
        model_save_path=os.path.join(os.path.dirname(__file__), 'models', 'tfno_to_respinn.pth'),
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'tfno_to_respinn.log'),
        lambda_data=1.0,
        lambda_phys=1e-5,
    )
    test_multitask_fpinn(
        model, tfno_teacher, test_loader,
        x_scaler=x_scaler, y_scaler=y_scaler,
        device=device,
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'tfno_to_respinn.log')
    )
    
    model = ResPINN(input_dim=21, embed_dim=32, hidden_dim=128, depth=6, output_dim=9)
    deeponet_teacher = DeepONet(
        branch_dim=21, trunk_dim=1, hidden_branch=128, hidden_trunk=128,
        p=128, output_dim=9, num_branch_layers=4, num_trunk_layers=4
    )
    deeponet_teacher.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'models', 'deeponet.pth')))
    deeponet_teacher.eval()
    train_multitask_fpinn(
        model, deeponet_teacher, train_loader, val_loader,
        num_epochs=500, lr=1e-3, device=device,
        x_scaler=x_scaler, y_scaler=y_scaler,
        model_save_path=os.path.join(os.path.dirname(__file__), 'models', 'deeponet_to_respinn.pth'),
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'deeponet_to_respinn.log'),
        lambda_data=1.0,
        lambda_phys=1e-5,
        teacher_model="deeponet"
    )
    test_multitask_fpinn(
        model, deeponet_teacher, test_loader,
        x_scaler=x_scaler, y_scaler=y_scaler,
        device=device,
        log_path=os.path.join(os.path.dirname(__file__), 'logs', 'deeponet_to_respinn.log')
    )

if __name__ == "__main__":
    train_dir = os.path.join(os.path.dirname(__file__), "data", "train")
    train_file_list = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir) if fname.endswith('.csv')]
    train_dataset = TBPDataset(train_file_list)  # For testing, limit to 100 files
    train_ds, val_ds, _ = random_split(train_dataset, [0.8, 0.2, 0], generator=torch.Generator().manual_seed(42))
    print(f"Loaded train dataset with {len(train_dataset)} samples.")
    
    test_dir = os.path.join(os.path.dirname(__file__), "data", "test")
    test_file_list = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith('.csv')]
    test_ds = TBPDataset(test_file_list)
    print(f"Loaded test dataset with {len(test_ds)} samples.")
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_scaler, y_scaler = compute_scalers(train_loader, device)
    run_multitask_deeponet_pipeline(train_loader, val_loader, test_loader, x_scaler, y_scaler)