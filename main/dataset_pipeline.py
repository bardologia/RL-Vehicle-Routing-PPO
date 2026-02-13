from core.dataset import Dataset
import ray # type: ignore
import os
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'

if ray.is_initialized():
    ray.shutdown()


ray.init(
    ignore_reinit_error=True,
    include_dashboard=False,
    log_to_driver=False,
    num_cpus=None,                 
    _metrics_export_port=None,
    object_store_memory=10*1024**3, 
    _temp_dir=None,                  
)

dataset = Dataset("datasets/chunked")

dataset.append(
    num_events=1024000,      
    output_dir="datasets/chunked",
    num_workers=8,        
    batch_size=128,      
    chunk_size=1024,      
    verbose=True,
    seed=42,
    enable_worker_profiling=True,
)
