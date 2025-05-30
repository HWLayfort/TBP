from PINN import run_pinn_pipeline
from FPINN import run_fpinn_pipeline
from ResPINN import run_respinn_pipeline
from FNO import run_fno_pipeline
from TFNO import run_tfno_pipeline
from DeepONet import run_deeponet_pipeline
from FNOtoPINN import run_fnotopinn_pipeline
from TFNOtoPINN import run_tfnotopinn_pipeline
from DeepONettoPINN import run_deeponettopinn_pipeline

def run_all_pipelines():
    print("Running FNO pipeline...")
    run_fno_pipeline()
    
    print("Running TFNO pipeline...")
    run_tfno_pipeline()
    
    print("Running DeepONet pipeline...")
    run_deeponet_pipeline()
    
    print("Running PINN pipeline...")
    run_pinn_pipeline()
    
    print("Running FPINN pipeline...")
    run_fpinn_pipeline()
    
    print("Running ResPINN pipeline...")
    run_respinn_pipeline()
    
def run_all_tranfer_pipelines():
    print("Running FNO to PINN transfer pipeline...")
    # run_fnotopinn_pipeline()
    
    print("Running TFNO to PINN transfer pipeline...")
    run_tfnotopinn_pipeline()
    
    print("Running DeepONet to PINN transfer pipeline...")
    # run_deeponettopinn_pipeline()
    
if __name__ == "__main__":
    # run_all_tranfer_pipelines()
    run_all_pipelines()