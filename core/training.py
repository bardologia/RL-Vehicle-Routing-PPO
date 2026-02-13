import torch.optim as optim
import gc
import math
import torch
from tools.logger import Logger, ModelSummary, TensorLogger
from tools.tracker import BaseTracker
from .environment import *
from .ppo import PPO
from tqdm import tqdm
import os


class LRScheduler:
    def __init__(self, optimizer, warmup_steps=1000, decay_steps=100000, lr_min=1e-5, writer=None):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.decay_steps  = decay_steps
        self.lr_min       = lr_min
        self.current_step = 0
        self.tracker   = BaseTracker(writer=writer) if writer else None
        
        self.lr_max_per_group = [pg['lr'] for pg in optimizer.param_groups]
      
    def step(self):
        self.current_step += 1
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.lr_max_per_group[i]
            
            if self.current_step < self.warmup_steps:
                progress = self.current_step / self.warmup_steps
                lr_start = 0.1 * base_lr
                lr = lr_start + (base_lr - lr_start) * progress
            else:
                adjusted_step = self.current_step - self.warmup_steps
                progress = min(1.0, adjusted_step / self.decay_steps)
                lr = self.lr_min + 0.5 * (base_lr - self.lr_min) * (1 + math.cos(math.pi * progress))
            
            param_group['lr'] = lr
        
        if self.tracker:
            self.tracker.log_optimizer(self.optimizer, self.current_step, prefix='batch/learning_rate')
        return self.optimizer.param_groups[0]['lr']
    
    def get_lr(self):
        return {f'group_{i}': pg['lr'] for i, pg in enumerate(self.optimizer.param_groups)}
    
    def set_step(self, step):
        self.current_step = step


class EntropyScheduler:
    def __init__(self, start_coef=0.02, end_coef=0.001, anneal_steps=50000, warmup_steps=0, writer=None):
        self.start_coef = start_coef
        self.end_coef = end_coef
        self.anneal_steps = anneal_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.tracker = BaseTracker(writer=writer) if writer else None
    
    def step(self):
        self.current_step += 1
        return self.get_coef()
    
    def get_coef(self):
        entropy_coef = self._linear_anneal()
        if self.tracker:
            self.tracker.log_scalar('batch/entropy_coefficient', entropy_coef, self.current_step)
        return entropy_coef

    def _linear_anneal(self):
        if self.current_step <= self.warmup_steps:
            return self.start_coef
        
        adjusted_step = self.current_step - self.warmup_steps
        progress = min(1.0, adjusted_step / self.anneal_steps)
        return self.start_coef + (self.end_coef - self.start_coef) * progress
    
    def set_step(self, step):
        self.current_step = step


class Trainer:
    def __init__(self, dataset=None, load_checkpoint=False, config=None):
        self.dataset         = dataset
        self.load_checkpoint = load_checkpoint
        self.config          = config

        self._clear_memory()
        self.device = config.training.device
        
        self.logger  = Logger(log_dir=config.io.logdir, name="training_log", level="INFO", config=config)
        self.tracker = BaseTracker(writer=config.io.writer)
        
        self.logger.section("[Trainer Initialization]")
        self.logger.subsection(f"Device: {self.device}")
        self.logger.subsection(f"Load Checkpoint: {load_checkpoint} \n")
        
        self.ppo           = self._initialize_ppo()
        self.environment   = Environment(config)
        self.summary       = ModelSummary(self.ppo)
        self.tensor_logger = TensorLogger(self.ppo).attach()
        self.attached      = True

        self.global_step_counter           = 0
        self.episode_index                 = 0
        self.ppo_update_index              = 0
        self.episode_start_index_in_memory = 0
        
        self.operator_stats = {'count': {i: 0 for i in range(4)}, 'rewards': {i: [] for i in range(4)}}
        
        self.logger.info("Trainer initialized successfully")

    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    def _log_gpu_memory(self, prefix=''):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)    # GB
            max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)  # GB
            
            self.tracker.log_scalar(f'batch/{prefix}_gpu_memory_allocated_gb', allocated, self.global_step_counter)
            self.tracker.log_scalar(f'batch/{prefix}_gpu_memory_reserved_gb', reserved, self.global_step_counter)
            self.tracker.log_scalar(f'batch/{prefix}_gpu_memory_max_allocated_gb', max_allocated, self.global_step_counter)
    
    def _initialize_ppo(self):
        self.logger.section("[PPO Initialization]")
        lr_cfg      = self.config.lr
        entropy_cfg = self.config.entropy
        io_cfg      = self.config.io
        
        self.logger.info("Creating PPO model")
        ppo = PPO(optimizer=None, config=self.config).to(self.device)
        ppo.tracker = BaseTracker(writer=io_cfg.writer)
        ppo.logger = self.logger
      
        param_groups = [
            {'params': ppo.policy.operator_actor.parameters(),       'lr': lr_cfg.lr_operator_actor,  'name': 'operator_actor'},
            {'params': ppo.policy.vehicle_actor.parameters(),        'lr': lr_cfg.lr_vehicle_actor,   'name': 'vehicle_actor'},
            {'params': ppo.policy.critic.parameters(),               'lr': lr_cfg.lr_critic,          'name': 'critic'},
            {'params': ppo.policy.graph_embedder_actor.parameters(), 'lr': lr_cfg.lr_embedder_actor,  'name': 'embedder_actor'},
            {'params': ppo.policy.job_pointer.parameters(),          'lr': lr_cfg.lr_job_pointer,     'name': 'job_pointer'},
            {'params': ppo.policy.pointer_context_proj.parameters(), 'lr': lr_cfg.lr_pointer_context, 'name': 'pointer_context'},
        ]
        
        self.logger.subsection(f"[LR] Scheduler Config: Warmup Steps={lr_cfg.lr_warmup_steps}, Decay Steps={lr_cfg.lr_decay_steps}, Min LR={lr_cfg.lr_min}")
        self.logger.subsubsection(f"[LR] Operator Actor={lr_cfg.lr_operator_actor}")
        self.logger.subsubsection(f"[LR] Vehicle Actor={lr_cfg.lr_vehicle_actor}")
        self.logger.subsubsection(f"[LR] Critic={lr_cfg.lr_critic}")
        self.logger.subsubsection(f"[LR] Embedder Actor={lr_cfg.lr_embedder_actor}")
        self.logger.subsubsection(f"[LR] Job Pointer={lr_cfg.lr_job_pointer}")
        self.logger.subsubsection(f"[LR] Pointer Context={lr_cfg.lr_pointer_context}")
        self.logger.subsubsection(f"[LR] Warmup Steps={lr_cfg.lr_warmup_steps} \n")
        
        self.logger.subsection(f"[Entropy] Scheduler Config")
        self.logger.subsubsection(f"Start Coef={entropy_cfg.entropy_start}")
        self.logger.subsubsection(f"End Coef={entropy_cfg.entropy_end}")
        self.logger.subsubsection(f"Anneal Steps={entropy_cfg.entropy_anneal_steps}")
                                
        optimizer = optim.Adam(param_groups, eps=1e-5)
        ppo.optimizer = optimizer
        
        ppo.lr_scheduler = LRScheduler(
            optimizer=optimizer,
            warmup_steps=lr_cfg.lr_warmup_steps,
            decay_steps=lr_cfg.lr_decay_steps,
            lr_min=lr_cfg.lr_min,
            writer=io_cfg.writer,
        )
        
        ppo.entropy_scheduler = EntropyScheduler(
            start_coef=entropy_cfg.entropy_start,
            end_coef=entropy_cfg.entropy_end,
            anneal_steps=entropy_cfg.entropy_anneal_steps,
            warmup_steps=lr_cfg.lr_warmup_steps,
            writer=io_cfg.writer,
        )
        
        ppo.current_entropy = entropy_cfg.entropy_start
        
        if self.load_checkpoint:
            self.logger.section("Loading Checkpoint")
            checkpoint_dir = io_cfg.logdir
            self.logger.subsection(f"Loading from directory: {checkpoint_dir}")
            training_state = ppo.policy.load(filename="graph_ppo_policy.pt", directory=checkpoint_dir)
           
            self.global_step_counter = training_state.get("global_step_counter", 0)
            self.episode_index       = training_state.get("episode_index", 0)
            self.ppo_update_index    = training_state.get("ppo_update_index", 0)
            
            self.logger.subsubsection(f"Restored Global Step: {self.global_step_counter}")
            self.logger.subsubsection(f"Restored Episode Index: {self.episode_index}")
            self.logger.subsubsection(f"Restored PPO Update Index: {self.ppo_update_index}")
            
            dataset_state = training_state.get("dataset_state", None)
            self.dataset.set_state(dataset_state)

            ppo.lr_scheduler.set_step(training_state["lr_scheduler_step"])
            ppo.entropy_scheduler.set_step(training_state["entropy_scheduler_step"])
            ppo.current_entropy = ppo.entropy_scheduler.get_coef()
            
            self.logger.subsubsection(f"LR Scheduler Step: {training_state['lr_scheduler_step']}")
            self.logger.subsubsection(f"Entropy Scheduler Step: {training_state['entropy_scheduler_step']}")
            self.logger.subsubsection(f"Current Entropy Coefficient: {ppo.current_entropy:.6f}")
               
        self.logger.subsubsection("PPO initialization complete")
        return ppo
    
    def _get_state(self):
        state = {
            "global_step_counter"    : self.global_step_counter,
            "episode_index"          : self.episode_index,
            "ppo_update_index"       : self.ppo_update_index,
            "dataset_state"          : self.dataset.get_state(),
            "lr_scheduler_step"      : self.ppo.lr_scheduler.current_step,
            "entropy_scheduler_step" : self.ppo.entropy_scheduler.current_step,
        }

        return state
    
    def ppo_update(self):
        self.logger.section("[PPO Update]")
        memory_size = len(self.ppo.memory.rewards)
        self.logger.info(f"Global Step: {self.global_step_counter}")
        self.logger.info(f"Buffer Size: {memory_size} samples")
        self.logger.info(f"PPO Update Index: {self.ppo_update_index}")
        self.tracker.log_scalar('batch/buffer_size', memory_size, self.global_step_counter)
        
        self.logger.subsection("GPU Memory Before Update")
        self._log_gpu_memory('before_update')
        
        self.logger.subsection("Running PPO Optimization")
        self.logger.info(f"Epochs: {self.config.training.num_epochs}, Minibatch Size: {self.config.training.minibatch_size}")
        self.ppo.update(number_of_epochs=self.config.training.num_epochs, minibatch_size=self.config.training.minibatch_size, global_step=self.global_step_counter)
        self.ppo_update_index += 1
        self.tracker.log_scalar('batch/ppo_updates', self.ppo_update_index, self.global_step_counter)
        self.logger.info(f"PPO update complete (Update #{self.ppo_update_index})")
        
        self.logger.subsection("GPU Memory After Update")
        self._log_gpu_memory('after_update')
         
        self.logger.subsection("Saving Checkpoint")
        training_state = self._get_state()
        self.ppo.checkpoint(filename="graph_ppo_policy.pt", directory=self.config.io.logdir, training_state=training_state)
        self.logger.info(f"Checkpoint saved to {self.config.io.logdir}")
        self.episode_start_index_in_memory = len(self.ppo.memory.rewards)
        
        self.logger.subsection("Clearing Memory")
        self._clear_memory()
        self._log_gpu_memory('after_clear')
        self.logger.info("Memory cleared successfully") 
    
    def run_episode(self, dataset_item):
        experiences = []
        self.environment.load_from_dataset(dataset_item)
        for step_in_episode in range(self.config.training.max_steps_per_episode):
            
            if step_in_episode == 0:
                graph     = dataset_item["graph"]
                mask_info = dataset_item["mask_info"]
            else:
                graph, mask_info = self.environment.observe()

            
            with torch.no_grad():
                if self.attached:
                    graph      = graph.to(self.device)
                    ppo_output = self.ppo.policy.act(graph, mask_info)
                    self.tensor_logger.save_markdown(path=os.path.join(self.config.io.logdir, "tensor_shape.md"), title=f"Tensor Shapes")
                    self.tensor_logger.detach()
                    self.attached = False
                    self.logger.section("[Tensor Logger] Tensor Shape Saved - Detaching Tensor Logger \n")
                else:
                    graph      = graph.to(self.device)
                    ppo_output = self.ppo.policy.act(graph, mask_info)

            true_vehicle_id = int(self.environment.vehicles[ppo_output["action"].vehicle_index]["id"])
            true_job_id     = int(self.environment.jobs[ppo_output["action"].job_index]["id"])

            old_state, next_state = self.environment.apply_action(ppo_output["action"])
            if next_state is None:
                break

            rewards, old_costs, new_costs = self.environment.step(old_state, next_state, ppo_output["action"].operator)
            reward = sum(rewards.values())
            op_id = ppo_output["action"].operator
            self.operator_stats['count'][op_id] += 1
            self.operator_stats['rewards'][op_id].append(reward)

            action_dict = {'operator': op_id, 'vehicle': ppo_output["action"].vehicle_index, 'job': ppo_output["action"].job_index}
            self.tracker.log_dict('step/action', action_dict, self.global_step_counter)
            self.tracker.log_dict('step/reward', rewards, self.global_step_counter)
            self.tracker.log_comparison('step/cost', old_costs, new_costs, self.global_step_counter, 'old', 'new')
            self.tracker.log_scalar('step/state_value', ppo_output["state_value"].item(), self.global_step_counter)
            
            attn_log = {
                'pointer_entropy'    : ppo_output['pointer_entropy'].item(),
                'glimpse_entropy'    : ppo_output['glimpse_entropy'].item(),
                'pointer_max_weight' : ppo_output['pointer_max_weight'].item(),
                'glimpse_max_weight' : ppo_output['glimpse_max_weight'].item(),
            }
            self.tracker.log_dict('step/pointer_attention', attn_log, self.global_step_counter)
            
            experience = {
                "graph": graph,
                "action": ppo_output["action"],
                "prob_operator": ppo_output["prob_operator"].detach().cpu(),
                "prob_vehicle": ppo_output["prob_vehicle"].detach().cpu(),
                "prob_job": ppo_output["prob_job"].detach().cpu(),
                "reward": float(reward),
                "state_value": ppo_output["state_value"].detach().cpu(),
                "mask_info": mask_info,
                "done": False,
                "old_operator_logits": ppo_output["old_operator_logits"].detach().cpu(),
                "old_vehicle_logits_by_operator": ppo_output["old_vehicle_logits_by_operator"].detach().cpu(),
                "old_job_logits_by_operator_vehicle": ppo_output["old_job_logits_by_operator_vehicle"].detach().cpu(),
                "true_vehicle_id": true_vehicle_id,
                "true_job_id": true_job_id,
            }
            
            experiences.append(experience)
  
        if experiences:
            experiences[-1]["done"] = True

        return experiences

    def run_chunk(self, chunk_data):
        self.logger.section("[Chunk Processing]")
        episodes_processed = 0
        self.logger.subsection(f"Total episodes in chunk: {len(chunk_data)}")
        self.logger.subsection(f"Starting episode index: {self.episode_index}")
        self._log_gpu_memory('start')
        
        for dataset_item in tqdm(chunk_data, desc="Processing episodes in chunk", leave=False):
            experiences = self.run_episode(dataset_item)
            
            episode_reward = sum([exp['reward'] for exp in experiences])
            episode_length = len(experiences)
            
            for experience in tqdm(experiences, desc="Adding experiences to memory", leave=False):
                self.ppo.memory.add(**experience)
                self.global_step_counter += 1

            episode_end_index = len(self.ppo.memory.rewards) - 1
            if episode_end_index >= self.episode_start_index_in_memory:
                self.ppo.memory.dones[episode_end_index] = True

            self.episode_index += 1
            episodes_processed += 1
            
            self.tracker.log_scalar('episode/total_reward', episode_reward, self.episode_index)
            self.tracker.log_scalar('episode/length', episode_length, self.episode_index)
            
            total = sum(self.operator_stats['count'].values())
            freq = {f'op{i}': self.operator_stats['count'][i]/total for i in range(4)}
            avg_rew = {f'op{i}': (sum(self.operator_stats['rewards'][i])/len(self.operator_stats['rewards'][i]) if self.operator_stats['rewards'][i] else 0) for i in range(4)}
            self.tracker.log_dict('episode/operator_frequency', freq, self.episode_index)
            self.tracker.log_dict('episode/operator_avg_reward', avg_rew, self.episode_index)
            self.operator_stats = {'count': {i: 0 for i in range(4)}, 'rewards': {i: [] for i in range(4)}}

        self.logger.subsubsection(f"Chunk complete: {episodes_processed} episodes processed")
        return episodes_processed
     
    def train(self):
        self.logger.section("[Training Start]")
        self.logger.subsection(f"Device Name: {torch.cuda.get_device_name(0)}")
        self.logger.subsection(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        self.logger.subsection(f"CUDA Version: {torch.version.cuda} \n")

        self.logger.section("[Output Directory]")
        self.logger.section(f"Log Directory: {self.config.io.logdir} \n")
        
        self.logger.section("Model Summary")
        self.logger.subsection("Generating model architecture summary")
        self.summary.run()
        self.summary.save_markdown(os.path.join(self.config.io.logdir, "model_summary.md"), title="PPO Model Summary")
        self.logger.subsubsection("Model summary saved \n")
        
        self.logger.section("[Training Loop]")
        chunk_paths = self.dataset.chunk_paths
        total_chunks = len(chunk_paths)
        self.logger.subsection(f"Total chunks to process: {total_chunks}")

        for chunk_idx, chunk_path in tqdm(enumerate(chunk_paths), desc="Training chunks", total=total_chunks):
            chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)

            self.episode_start_index_in_memory = len(self.ppo.memory.rewards)
            episodes_processed = self.run_chunk(chunk_data)
            
            self.tracker.log_scalar('batch/episodes_processed', episodes_processed, chunk_idx)
            
            self.ppo_update()
            
            progress_pct = ((chunk_idx + 1) / total_chunks) * 100
            self.logger.info(f"Overall Progress: {progress_pct:.1f}% ({chunk_idx + 1}/{total_chunks} chunks)")
            self.tracker.log_scalar('batch/chunk_progress', chunk_idx / total_chunks, chunk_idx)
            
            self.logger.subsection("Cleanup")
            del chunk_data
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.info("Chunk data cleared")
        
        self.logger.section("[Training Complete]")
        self.logger.subsection("Saving final checkpoint")
        training_state = self._get_state()
        self.ppo.policy.checkpoint(filename="graph_ppo_policy.pt", directory=self.config.io.logdir, training_state=training_state)
        self.ppo.memory.clear()
        self.logger.subsubsection("Training finished successfully")
        return self.ppo