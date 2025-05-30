# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# from tqdm import tqdm
# from TFNO import TFNO
# from PINN import PINN
# from FPINN import FPINN
# from ResPINN import ResPINN

# # FNO trajectory 예측 결과를 단일 시점 supervision용으로 변환
# class TFNOSingleStepDistillDataset(torch.utils.data.Dataset):
#     def __init__(self, filelist, teacher_model, device='cpu'):
#         # teacher_model은 FNO (trajectory predictor)
#         self.data = []
#         teacher_model.eval()
#         teacher_model = teacher_model.to(device)
#         for fname in filelist:
#             filename = os.path.basename(fname)
#             base = filename.split('_')[0]
#             mass_str = base.replace('m', '')
#             masses = np.array([float(x) for x in mass_str.split('-')])
#             df = pd.read_csv(fname, header=None)
#             t_arr = df.iloc[:, 0].values          # (N,)
#             N = t_arr.shape[0]
#             r0 = df.iloc[0, 1:10].values      # (9,)
#             v0 = df.iloc[0, 10:19].values     # (9,)
#             m = masses
#             # input trajectory: (22, N)
#             r0_mat = np.tile(r0[:, None], (1, N))  # (9, N)
#             v0_mat = np.tile(v0[:, None], (1, N))  # (9, N)
#             m_mat  = np.tile(m[:, None], (1, N))   # (3, N)
#             t_mat  = t_arr[None, :]                # (1, N)
#             x_traj = np.concatenate([r0_mat, v0_mat, m_mat, t_mat], axis=0)   # (22, N)

#             # torch tensor for FNO inference
#             x_traj_tensor = torch.tensor(x_traj[None, ...], dtype=torch.float32, device=device)  # (1, 22, N)
#             with torch.no_grad():
#                 y_traj_pred = teacher_model(x_traj_tensor)   # (1, 9, N)
#             y_traj_pred = y_traj_pred.cpu().numpy().squeeze()  # (9, N)
#             # 단일 시점별로 sample로 쪼갬
#             for k in range(N):
#                 # PINN/ResPINN 입력 포맷과 동일하게
#                 x_single = np.concatenate([
#                     np.array([t_arr[k]]),    # t
#                     r0,                      # r0 (9,)
#                     v0,                      # v0 (9,)
#                     m                       # mass (3,)
#                 ])   # shape (22,)
#                 y_single = y_traj_pred[:, k] # shape (9,)
#                 self.data.append((x_single, y_single, m))
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, idx):
#         x, y, m = self.data[idx]
#         return (
#             torch.tensor(x, dtype=torch.float32),
#             torch.tensor(y, dtype=torch.float32),
#             torch.tensor(m, dtype=torch.float32)
#         )

# def compute_physics_residual(r_pred, a_pred, masses, G=1.0, eps=1e-8):
#     B = r_pred.shape[0]
#     r = r_pred.view(B, 3, 3)
#     a = a_pred.view(B, 3, 3)
#     m = masses.view(B, 1, 3, 1)
#     diff = r.unsqueeze(2) - r.unsqueeze(1)
#     norm = torch.norm(diff, dim=-1, keepdim=True) + eps
#     a_grav = G * m * diff / (norm ** 3)
#     mask = 1 - torch.eye(3, device=r.device).view(1, 3, 3, 1)
#     a_grav = (a_grav * mask).sum(dim=2)
#     return (a - a_grav).view(B, -1)

# def compute_acceleration(model, x, eps=1e-3):
#     # x: (batch, input_dim)
#     def shift_t(x, delta):
#         x_new = x.clone()
#         x_new[:, 0] = x[:, 0] + delta
#         return x_new

#     r      = model(x)
#     r_p1   = model(shift_t(x, +eps))
#     r_m1   = model(shift_t(x, -eps))
#     r_p2   = model(shift_t(x, +2*eps))
#     r_m2   = model(shift_t(x, -2*eps))

#     a = (-r_m2 + 16*r_m1 - 30*r + 16*r_p1 - r_p2) / (12 * eps**2)
#     return a

# def test_only_loop(model, test_loader, device, desc="Test (MSE only)"):
#     """
#     ResPINN의 테스트 단계에서 오직 예측값(y_pred)과 라벨(yb)의 MSE만 계산하는 함수.
#     distillation, physics loss 등은 포함하지 않는다.
#     """
#     model.eval()
#     loss_fn = nn.MSELoss()
#     total_loss = 0.0
#     n = 0
#     with torch.no_grad():
#         for xb, yb, mb in tqdm(test_loader, desc=desc):
#             xb = xb.to(device)
#             yb = yb.to(device)
#             # mb = mb.to(device)  # masses, 사용하지 않음
#             y_pred = model(xb)
#             loss = loss_fn(y_pred, yb)
#             total_loss += loss.item() * xb.size(0)
#             n += xb.size(0)
#     avg_loss = total_loss / n
#     return avg_loss


# def fno_predict_single_step(teacher_model, xb):
#     """
#     teacher_model: FNO, expects (batch, input_dim, N)
#     xb: (batch, input_dim)
#     Returns: (batch, output_dim)
#     """
#     # xb: (batch, input_dim)
#     # xb의 t, r0, v0, m에서 t만 다르고 나머지는 고정이므로, trajectory로 재구성
#     t_val = xb[:, 0]         # (batch,)
#     r0 = xb[:, 1:10]         # (batch, 9)
#     v0 = xb[:, 10:19]        # (batch, 9)
#     m  = xb[:, 19:22]        # (batch, 3)
#     # trajectory 길이 N: 실제 trajectory 생성에 사용한 길이
#     N = 10000  # ← 생성 데이터셋의 N (미리 고정 또는 xb에서 알 수 있으면 동적으로)
#     # 모든 t 후보 배열 (dataset과 동일하게!)
#     t_all = np.linspace(0, 10, N, dtype=np.float32)
#     x_teacher = []
#     for b in range(xb.size(0)):
#         r0_mat = r0[b].cpu().numpy().reshape(9, 1).repeat(N, axis=1)
#         v0_mat = v0[b].cpu().numpy().reshape(9, 1).repeat(N, axis=1)
#         m_mat  = m[b].cpu().numpy().reshape(3, 1).repeat(N, axis=1)
#         t_mat  = t_all.reshape(1, -1)
#         x_traj = np.concatenate([r0_mat, v0_mat, m_mat, t_mat], axis=0)  # (22, N)
#         x_teacher.append(x_traj)
#     x_teacher = np.stack(x_teacher, axis=0)   # (batch, 22, N)
#     x_teacher = torch.tensor(x_teacher, dtype=torch.float32, device=xb.device)
#     with torch.no_grad():
#         y_traj = teacher_model(x_teacher)   # (batch, 9, N)
#     # t와 t_all을 비교해서 각 샘플의 t에 가장 가까운 idx만 뽑기
#     idxs = [np.abs(t_all - float(tv.cpu())).argmin() for tv in t_val]
#     y_teacher = torch.stack([y_traj[i, :, idxs[i]] for i in range(len(idxs))], dim=0)  # (batch, 9)
#     return y_teacher

# def train_single_step_distill(
#     teacher_model, student_model, 
#     train_loader, val_loader, test_loader, 
#     num_epochs=100, lr=1e-3, device='cuda',
#     lambda_phys=1.0, lambda_distill=1.0,
#     model_save_path='respinn_distill_step.pth', 
#     log_path=None,
#     early_stop_patience=20
# ):
#     teacher_model.eval()
#     student_model = student_model.to(device)
#     teacher_model = teacher_model.to(device)
#     optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()
#     best_val = float('inf')
#     patience = 0
    
#     if log_path is not None:
#         with open(log_path, "w") as f:
#             f.write("epoch,train_loss,train_distill_loss,train_phys_loss,val_loss,val_distill_loss,val_phys_loss\n")
    
#     for epoch in range(num_epochs):
#         student_model.train()
#         total_loss, total_distill, total_phys, n = 0.0, 0.0, 0.0, 0
#         for xb, yb, mb in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
#             xb = xb.to(device)  # (batch, input_dim)
#             yb = yb.to(device)  # (batch, output_dim)
#             mb = mb.to(device)
#             optimizer.zero_grad()
#             # distillation loss
#             with torch.no_grad():
#                 y_teacher = fno_predict_single_step(teacher_model, xb)
#             y_student = student_model(xb)
#             loss_distill = loss_fn(y_student, y_teacher)
#             # physics loss
#             xb_req = xb.clone().detach()
#             r_pred = student_model(xb_req)
#             a_pred = compute_acceleration(student_model, xb_req)
#             loss_phys = (compute_physics_residual(r_pred, a_pred, mb)**2).mean()
#             loss = lambda_distill * loss_distill + lambda_phys * loss_phys
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * xb.size(0)
#             total_distill += loss_distill.item() * xb.size(0)
#             total_phys += loss_phys.item() * xb.size(0)
#             n += xb.size(0)
#         avg_loss = total_loss / n
#         avg_distill = total_distill / n
#         avg_phys = total_phys / n
#         val_loss, val_distill, val_phys = eval_single_step_distill(
#             teacher_model, student_model, val_loader, device, loss_fn, lambda_phys, lambda_distill
#         )
        
#         if log_path is not None:
#             with open(log_path, "a") as f:
#                 f.write(f"{epoch},{avg_loss},{avg_loss},{avg_phys},{val_loss},{val_distill},{val_phys}\n")
        
#         if val_loss < best_val:
#             best_val = val_loss
#             patience = 0
#             torch.save(student_model.state_dict(), model_save_path)
#         else:
#             patience += 1
#             if patience >= early_stop_patience:
#                 print(f"Early stopping at epoch {epoch+1}")
#                 break
#         if epoch % 10 == 0:
#             print(f"[Epoch {epoch+1}] Loss: {avg_loss:.6f} (distill: {avg_distill:.6f}, phys: {avg_phys:.6f}) | Val Loss: {val_loss:.6f}")
            
#     test_mse_loss = test_only_loop(student_model, test_loader, device)
#     print(f"Final Test MSE Loss: {test_mse_loss:.6f}")
#     if log_path is not None:
#         with open(log_path, "a") as f:
#             f.write(f"test,{test_mse_loss}\n")

# def eval_single_step_distill(
#     teacher_model, student_model, loader, device, loss_fn, lambda_phys, lambda_distill
# ):
#     teacher_model.eval()
#     student_model.eval()
#     total_loss, total_distill, total_phys, n = 0.0, 0.0, 0.0, 0
#     for xb, yb, mb in loader:
#         xb = xb.to(device)
#         yb = yb.to(device)
#         mb = mb.to(device)
#         with torch.no_grad():
#             y_teacher = fno_predict_single_step(teacher_model, xb)
#             y_student = student_model(xb)
#             loss_distill = loss_fn(y_student, y_teacher)
#             xb_req = xb.clone().detach()
#             r_pred = student_model(xb_req)
#             a_pred = compute_acceleration(student_model, xb_req)
#             loss_phys = (compute_physics_residual(r_pred, a_pred, mb)**2).mean()
#             loss = lambda_distill * loss_distill + lambda_phys * loss_phys
#         total_loss += loss.item() * xb.size(0)
#         total_distill += loss_distill.item() * xb.size(0)
#         total_phys += loss_phys.item() * xb.size(0)
#         n += xb.size(0)
#     return total_loss/n, total_distill/n, total_phys/n

# def transfer_train(target_model, source_model, 
#                    data_dir=None, 
#                    model_save_path='respinn_distill_step.pth', 
#                    log_path=None,
#                    num_epochs=100, 
#                    batch_size=1024, 
#                    lambda_phys=1.0, 
#                    lambda_distill=1.0, 
#                    early_stop_patience=20, 
#                    device=None):
#     # 장치 설정
#     if device is None:
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     source_model = source_model.to(device)
#     target_model = target_model.to(device)

#     # 데이터 경로
#     if data_dir is None:
#         data_dir = os.path.join(os.path.dirname(__file__), "data", "output")
#     file_list = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.csv')]

#     # FNO Trajectory → 단일 시점 distillation dataset 생성
#     distill_dataset = TFNOSingleStepDistillDataset(file_list, source_model, device=device)
#     print(f"Loaded dataset with {len(distill_dataset)} single-step samples.")
#     total = len(distill_dataset)
#     n_train = int(0.7 * total)
#     n_val = int(0.15 * total)
#     n_test = total - n_train - n_val
#     train_ds, val_ds, test_ds = random_split(distill_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

#     # Knowledge Distillation + Physics Loss 학습
#     train_single_step_distill(
#         source_model, target_model,
#         train_loader, val_loader, test_loader,
#         num_epochs=num_epochs, lr=1e-3, device=device,
#         lambda_phys=lambda_phys, lambda_distill=lambda_distill,
#         model_save_path=model_save_path, 
#         log_path=log_path,
#         early_stop_patience=early_stop_patience
#     )

# def run_tfnotopinn_pipeline():
#     tfno_teacher = TFNO(
#         t_steps=10000, input_dim=22, embed_dim=32, width=64, modes=32,
#         depth=4, output_dim=9
#     )
#     tfno_teacher.load_state_dict(torch.load('tfno.pth', map_location='cpu'))
    
#     pinn_student = PINN(input_dim=22, hidden_dim=128, num_hidden_layers=8, output_dim=9)
#     fpinn_student = FPINN(mapping_size=32, scale=10.0, hidden_dim=128, num_hidden_layers=8, output_dim=9)
#     respinn_student = ResPINN(input_dim=22, hidden_dim=128, num_hidden_layers=8, output_dim=9)
    
#     # Knowledge Distillation: FNO, TFNO, DeepONet → PINN
#     transfer_train(pinn_student, tfno_teacher, data_dir='data/output', num_epochs=500, model_save_path="tfno_to_pinn.pth", log_path='tfno_to_pinn.log')
#     transfer_train(fpinn_student, tfno_teacher, data_dir='data/output', num_epochs=500, model_save_path="tfno_to_fpinn.pth", log_path='tfno_to_fpinn.log')
#     transfer_train(respinn_student, tfno_teacher, data_dir='data/output', num_epochs=500, model_save_path="tfno_to_respinn.pth", log_path='tfno_to_respinn.log')
    
# if __name__ == "__main__":
#     run_tfnotopinn_pipeline()
#     print("TFNO to PINN transfer learning completed.")


# Optimized TFNO to PINN knowledge distillation pipeline
# Major improvements:
# 1. Precompute teacher (TFNO) predictions to avoid repeated inference
# 2. Simplified acceleration (3-point stencil)
# 3. AMP (mixed precision)
# 4. Cached physics mask

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from TFNO import TFNO
from PINN import PINN
from FPINN import FPINN
from ResPINN import ResPINN

class CachedTFNOData(torch.utils.data.Dataset):
    def __init__(self, filelist, teacher_model, device='cpu'):
        self.data = []
        teacher_model.eval()
        teacher_model = teacher_model.to(device)
        for fname in filelist:
            filename = os.path.basename(fname)
            base = filename.split('_')[0]
            masses = np.array([float(x) for x in base.replace('m', '').split('-')])
            df = pd.read_csv(fname, header=None)
            t_arr = df.iloc[:, 0].values
            N = len(t_arr)
            r0 = df.iloc[0, 1:10].values
            v0 = df.iloc[0, 10:19].values
            r0_mat = np.tile(r0[:, None], (1, N))
            v0_mat = np.tile(v0[:, None], (1, N))
            m_mat = np.tile(masses[:, None], (1, N))
            t_mat = t_arr[None, :]
            x_traj = np.concatenate([r0_mat, v0_mat, m_mat, t_mat], axis=0)
            x_traj_tensor = torch.tensor(x_traj[None, ...], dtype=torch.float32, device=device)
            with torch.no_grad():
                y_traj_pred = teacher_model(x_traj_tensor).cpu().numpy().squeeze()
            for k in range(N):
                x_single = np.concatenate([
                    [t_arr[k]], r0, v0, masses
                ])
                y_single = y_traj_pred[:, k]
                self.data.append((x_single, y_single, masses))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, m = self.data[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(m, dtype=torch.float32)
        )

def compute_acceleration(model, x, eps=1e-3):
    def shift_t(x, delta):
        x_new = x.clone()
        x_new[:, 0] += delta
        return x_new
    r = model(x)
    r_p1 = model(shift_t(x, eps))
    r_m1 = model(shift_t(x, -eps))
    a = (r_p1 - 2 * r + r_m1) / (eps ** 2)
    return a

def compute_physics_residual(r_pred, a_pred, masses, mask, G=1.0, eps=1e-8):
    B = r_pred.shape[0]
    r = r_pred.view(B, 3, 3)
    a = a_pred.view(B, 3, 3)
    m = masses.view(B, 1, 3, 1)
    diff = r.unsqueeze(2) - r.unsqueeze(1)
    norm = torch.norm(diff, dim=-1, keepdim=True) + eps
    a_grav = G * m * diff / (norm ** 3)
    a_grav = (a_grav * mask).sum(dim=2)
    return (a - a_grav).view(B, -1)

def test_only_loop(model, test_loader, device, desc="Test (MSE only)"):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb, mb in tqdm(test_loader, desc=desc):
            xb = xb.to(device)
            yb = yb.to(device)
            # mb = mb.to(device)  # masses, 사용하지 않음
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    avg_loss = total_loss / n
    return avg_loss

def train_distill(teacher_model, student_model, train_loader, val_loader, test_loader,
                  num_epochs=100, lr=1e-3, device='cuda', lambda_phys=1.0, lambda_distill=1.0,
                  model_save_path='student.pth', log_path=None, early_stop_patience=20):

    teacher_model.eval()
    student_model.to(device)
    teacher_model.to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = GradScaler(device)
    mask = 1 - torch.eye(3, device=device).view(1, 3, 3, 1)

    best_val = float('inf')
    patience = 0
    
    if log_path is not None:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_distill_loss,train_phys_loss,val_loss,val_distill_loss,val_phys_loss\n")

    for epoch in range(num_epochs):
        student_model.train()
        total_loss, total_distill, total_phys, n = 0.0, 0.0, 0.0, 0
        for xb, yb, mb in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device):
                y_pred = student_model(xb)
                loss_distill = loss_fn(y_pred, yb)
                a_pred = compute_acceleration(student_model, xb)
                loss_phys = (compute_physics_residual(y_pred, a_pred, mb, mask) ** 2).mean()
                loss = lambda_distill * loss_distill + lambda_phys * loss_phys
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * xb.size(0)
            total_distill += loss_distill.item() * xb.size(0)
            total_phys += loss_phys.item() * xb.size(0)
            n += xb.size(0)
        
        avg_loss = total_loss / n
        avg_distill = total_distill / n
        avg_phys = total_phys / n

        val_loss, val_distill, val_phys = validate(student_model, val_loader, device, loss_fn, mask, lambda_phys, lambda_distill)
        
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(f"{epoch},{avg_loss},{avg_distill},{avg_phys},{val_loss},{val_distill},{val_phys}\n")
        
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(student_model.state_dict(), model_save_path)
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    test_loss = test_only_loop(student_model, test_loader, device)
    print(f"Final Test Loss: {test_loss:.6f}")
    
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"test,{test_loss}\n")

def validate(model, loader, device, loss_fn, mask, lambda_phys, lambda_distill):
    model.eval()
    total_loss, total_distill, total_phys, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            y_pred = model(xb)
            loss_distill = loss_fn(y_pred, yb)
            a_pred = compute_acceleration(model, xb)
            loss_phys = (compute_physics_residual(y_pred, a_pred, mb, mask) ** 2).mean()
            loss = lambda_distill * loss_distill + lambda_phys * loss_phys
            total_distill += loss_distill.item() * xb.size(0)
            total_phys += loss_phys.item() * xb.size(0)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    return total_loss / n, total_distill / n, total_phys / n

def run_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tfno_teacher = TFNO(t_steps=10000, input_dim=22, embed_dim=32, width=64, modes=32, depth=4, output_dim=9)
    tfno_teacher.load_state_dict(torch.load('tfno.pth', map_location=device))

    students = {
        'PINN': PINN(input_dim=22, hidden_dim=128, num_hidden_layers=8, output_dim=9),
        'FPINN': FPINN(mapping_size=32, scale=10.0, hidden_dim=128, num_hidden_layers=8, output_dim=9),
        'ResPINN': ResPINN(input_dim=22, hidden_dim=128, num_hidden_layers=8, output_dim=9),
    }

    data_dir = 'data/output'
    file_list = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.csv')]
    dataset = CachedTFNOData(file_list, tfno_teacher, device=device)
    n = len(dataset)
    train_ds, val_ds, test_ds = random_split(dataset, [int(n*0.7), int(n*0.15), n - int(n*0.85)])
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)
    test_loader = DataLoader(test_ds, batch_size=1024)

    for name, student_model in students.items():
        print(f"Training {name}...")
        train_distill(tfno_teacher, student_model, train_loader, val_loader, test_loader,
                      model_save_path=f'tfno_to_{name.lower()}.pth', log_path=f'tfno_to_{name.lower()}.log',
                      lambda_phys=1.0, lambda_distill=1.0,
                      device=device)

if __name__ == '__main__':
    run_pipeline()