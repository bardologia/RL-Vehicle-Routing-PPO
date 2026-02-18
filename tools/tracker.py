import torch
from torch.utils.tensorboard import SummaryWriter


class BaseTracker:
    def __init__(self, writer):
        self.writer = writer
    
    def log_scalar(self, name: str, value, step: int):
        val = value.item() if hasattr(value, 'item') else value
        if self.writer:
            self.writer.add_scalar(name, val, step)
    
    def log_dict(self, prefix: str, data_dict: dict, step: int, add_comparison: bool = True):
        comparison_dict = {}
        for key, value in data_dict.items():
            val = value.item() if hasattr(value, 'item') else value
            if self.writer:
                self.writer.add_scalar(f'{prefix}/{key}', val, step)
            comparison_dict[key] = val
        
        if add_comparison and len(comparison_dict) > 1:
            if self.writer:
                self.writer.add_scalars(f'{prefix}/comparison', comparison_dict, step)
    
    def log_comparison(self, prefix: str, dict1: dict, dict2: dict, step: int, 
                      label1: str = 'before', label2: str = 'after'):

        self.log_dict(f'{prefix}/{label1}', dict1, step, add_comparison=True)
        self.log_dict(f'{prefix}/{label2}', dict2, step, add_comparison=True)
        
        if set(dict1.keys()) == set(dict2.keys()):
            delta_dict = {}
            for key in dict1.keys():
                v1 = dict1[key].item() if hasattr(dict1[key], 'item') else dict1[key]
                v2 = dict2[key].item() if hasattr(dict2[key], 'item') else dict2[key]
                delta_dict[key] = v2 - v1
            
            if delta_dict:
                if self.writer:
                    self.writer.add_scalars(f'{prefix}/delta', delta_dict, step)
    
    def log_grouped_dict(self, prefix: str, data_dict: dict, step: int, group_patterns: list = None):
        for key, value in data_dict.items():
            val = value.item() if hasattr(value, 'item') else value
            if self.writer:
                self.writer.add_scalar(f'{prefix}/{key}', val, step)
        
        if group_patterns:
            for pattern in group_patterns:
                grouped = {}
                for key, value in data_dict.items():
                    if pattern in key:
                        val = value.item() if hasattr(value, 'item') else value
                        grouped[key] = val
                
                if len(grouped) > 1:
                    if self.writer:
                        self.writer.add_scalars(f'{prefix}/{pattern}_group', grouped, step)
    
    def log_split_dict(self, prefix: str, data_dict: dict, step: int, prefixes: tuple):
        for key, value in data_dict.items():
            val = value.item() if hasattr(value, 'item') else value
            if self.writer:
                self.writer.add_scalar(f'{prefix}/{key}', val, step)
        
        for group_prefix in prefixes:
            grouped = {}
            for key, value in data_dict.items():
                if key.startswith(group_prefix):
                    clean_key = key.replace(f'{group_prefix}_', '')
                    val = value.item() if hasattr(value, 'item') else value
                    grouped[clean_key] = val
            
            if grouped:
                if self.writer:
                    self.writer.add_scalars(f'{prefix}/{group_prefix}', grouped, step)
    
    def log_optimizer(self, optimizer, step: int, prefix: str = 'optimizer'):
        state_dict = {}
        for i, param_group in enumerate(optimizer.param_groups):
            component_name = param_group.get('name', f'group_{i}')
        
            lr = param_group['lr']
            if self.writer:
                self.writer.add_scalar(f'{prefix}/lr_{component_name}', lr, step)
            state_dict[f'lr_{component_name}'] = lr
            
            for key in ['momentum', 'weight_decay', 'eps']:
                if key in param_group:
                    val = param_group[key]
                    if self.writer:
                        self.writer.add_scalar(f'{prefix}/{key}_{component_name}', val, step)
                    state_dict[f'{key}_{component_name}'] = val
        
        if len(state_dict) > 1 and self.writer:
            self.writer.add_scalars(f'{prefix}/comparison', state_dict, step)
    
    def log_gpu_memory(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)    # GB
            max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)  # GB
            
            self.log_scalar(f'batch/gpu_memory_allocated_gb', allocated, self.global_step_counter)
            self.log_scalar(f'batch/gpu_memory_reserved_gb', reserved, self.global_step_counter)
            self.log_scalar(f'batch/gpu_memory_max_allocated_gb', max_allocated, self.global_step_counter)

    def close(self):
        if self.writer:
            self.writer.close()