from DataLoader import TBPDataset, compute_scalers
from PINN import run_pinn_pipeline
from FPINN import run_fpinn_pipeline
from ResPINN import run_respinn_pipeline
from FNO import run_fno_pipeline
from TFNO import run_tfno_pipeline
from DeepONet import run_deeponet_pipeline

import os
import torch
from torch.utils.data import DataLoader, random_split
# from FNOtoPINN import run_fnotopinn_pipeline
# from TFNOtoPINN import run_tfnotopinn_pipeline
# from DeepONettoPINN import run_deeponettopinn_pipeline

def run_all_pipelines():
    train_dir = os.path.join(os.path.dirname(__file__), "data", "train")
    train_file_list = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir) if fname.endswith('.csv')]
    train_dataset = TBPDataset(train_file_list)  # For testing, limit to 100 files
    train_ds, val_ds, _ = random_split(train_dataset, [0.8, 0.2, 0], generator=torch.Generator().manual_seed(42))
    print(f"Loaded train dataset with {len(train_dataset)} samples.")
    
    test_dir = os.path.join(os.path.dirname(__file__), "data", "test")
    test_file_list = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith('.csv')]
    test_ds = TBPDataset(test_file_list)
    print(f"Loaded test dataset with {len(test_ds)} samples.")
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_scaler, y_scaler = compute_scalers(train_loader, device)
    # print("Running FNO pipeline...")
    # run_fno_pipeline(
    #     train_loader, val_loader, test_loader, 
    #     x_scaler=x_scaler, y_scaler=y_scaler, device=device
    # )
    
    # print("Running TFNO pipeline...")
    # run_tfno_pipeline(
    #     train_loader, val_loader, test_loader, 
    #     x_scaler=x_scaler, y_scaler=y_scaler, device=device
    # )
    
    # print("Running DeepONet pipeline...")
    # run_deeponet_pipeline(
    #     train_loader, val_loader, test_loader, 
    #     x_scaler=x_scaler, y_scaler=y_scaler, device=device
    # )
    
    print("Running PINN pipeline...")
    run_pinn_pipeline(
        train_loader, val_loader, test_loader, 
        x_scaler=x_scaler, y_scaler=y_scaler, device=device
    )
    
    print("Running FPINN pipeline...")
    run_fpinn_pipeline(
        train_loader, val_loader, test_loader, 
        x_scaler=x_scaler, y_scaler=y_scaler, device=device
    )
    
    print("Running ResPINN pipeline...")
    run_respinn_pipeline(
        train_loader, val_loader, test_loader, 
        x_scaler=x_scaler, y_scaler=y_scaler, device=device
    )
    
# def run_all_tranfer_pipelines():
#     print("Running FNO to PINN transfer pipeline...")
#     # run_fnotopinn_pipeline()
    
#     print("Running TFNO to PINN transfer pipeline...")
#     run_tfnotopinn_pipeline()
    
#     print("Running DeepONet to PINN transfer pipeline...")
#     # run_deeponettopinn_pipeline()
    
if __name__ == "__main__":
    run_all_pipelines()