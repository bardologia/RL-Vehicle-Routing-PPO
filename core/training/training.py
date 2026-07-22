import torch.optim as optim
import gc
import torch
from tools.logger import Logger
from tools.inspection import ModelSummary, TensorLogger
from tools.tracker import Tracker
from tools.telemetry import PPOTelemetry
from core.shared import Environment, ActionMasker, vroom
from .ppo import PPO, ActionDistribution
from .schedulers import LRScheduler, EntropyScheduler, EpochEarlyStopping
from tqdm import tqdm
import os


class Checkpoint:
    def __init__(self, config, logger=None):
        self.config   = config
        self.logger   = logger
        self.filename = "graph_ppo_policy.pt"
    
    def load(self, ppo, trainer, dataset, directory=None):
        load_dir = directory or self.config.io.logdir
        
        self.logger.section("[Loading Checkpoint]")
        
        training_state = ppo.policy.load(
            filename=self.filename,
            directory=load_dir
        )
        
        # Load optimizer state if available
        checkpoint_path = os.path.join(load_dir, self.filename)
        checkpoint = torch.load(checkpoint_path, map_location=ppo.device, weights_only=False)
        if "optimizer_state_dict" in checkpoint:
            ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.logger.subsection("Optimizer state restored")
        
        trainer.global_step_counter = training_state.get("global_step_counter", 0)
        trainer.episode_index = training_state.get("episode_index", 0)
        trainer.ppo_update_index = training_state.get("ppo_update_index", 0)
        
        self.logger.subsection(f"Restored Global Step      : {trainer.global_step_counter}")
        self.logger.subsection(f"Restored Episode Index    : {trainer.episode_index}")
        self.logger.subsection(f"Restored PPO Update Index : {trainer.ppo_update_index}")

        dataset_state = training_state.get("dataset_state", None)
        dataset.set_state(dataset_state)
        
        ppo.lr_scheduler.set_step(training_state["lr_scheduler_step"])
        ppo.entropy_scheduler.set_step(training_state["entropy_scheduler_step"])
        ppo.current_entropy_coef = ppo.entropy_scheduler.get_coef()
        
        self.logger.subsection(f"LR Scheduler Step           : {training_state['lr_scheduler_step']}")
        self.logger.subsection(f"Entropy Scheduler Step      : {training_state['entropy_scheduler_step']}")
        self.logger.subsection(f"Current Entropy Coefficient : {ppo.current_entropy_coef:.6f} \n")

        return training_state

    def save(self, ppo, trainer, directory=None):
        save_dir = directory or self.config.io.logdir

        ppo.policy.checkpoint(
            filename       = self.filename,
            directory      = save_dir,
            training_state = trainer.state(),
            optimizer      = ppo.optimizer,
        )

        self.logger.subsection(f"Checkpoint saved to {os.path.join(save_dir, self.filename)}")


class Trainer:
    def __init__(self, dataset=None, load_checkpoint=False, config=None):
        self.dataset         = dataset
        self.load_checkpoint = load_checkpoint
        self.config          = config

        self.device     = config.training.device
        
        self.logger     = Logger(log_dir=config.io.logdir, name="training", level="INFO")
        self.tracker    = Tracker(writer=config.io.writer)
        self.telemetry  = PPOTelemetry(self.tracker, config)
        self.checkpoint = Checkpoint(config, logger=self.logger)

        vroom.logger = self.logger

        self.logger.section("[Trainer Initialization]")
        self.logger.subsection(f"Device: {self.device}")
        self.logger.subsection(f"Load Checkpoint: {load_checkpoint} \n")

        self.ppo           = self.initialize()
        self.environment   = Environment(config, logger=self.logger)
        self.summary       = ModelSummary(self.ppo)
        self.tensor_logger = TensorLogger(self.ppo).attach()
        self.attached      = True

        self.global_step_counter           = 0
        self.episode_index                 = 0
        self.ppo_update_index              = 0
        
        self.operator_stats = {'count': {i: 0 for i in range(4)}, 'rewards': {i: [] for i in range(4)}}
        
        self.clear_memory()
        self.logger.subsection("Trainer initialized successfully")

    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    def initialize(self):
        self.logger.section("[PPO Initialization]")
        lr_config      = self.config.lr
        entropy_config = self.config.entropy
        io_config      = self.config.io
                
        masker       = ActionMasker(self.config)
        distribution = ActionDistribution(self.config, masker)

        ppo = PPO(optimizer=None, config=self.config).to(self.device)

        ppo.telemetry    = self.telemetry
        ppo.logger       = self.logger
        ppo.masker       = masker
        ppo.distribution = distribution
      
        param_groups = [
            {'params': ppo.policy.operator_actor.parameters(),       'lr': lr_config.lr_operator_actor,  'name': 'operator_actor'},
            {'params': ppo.policy.vehicle_actor.parameters(),        'lr': lr_config.lr_vehicle_actor,   'name': 'vehicle_actor'},
            {'params': ppo.policy.critic.parameters(),               'lr': lr_config.lr_critic,          'name': 'critic'},
            {'params': ppo.policy.graph_embedding.parameters(),      'lr': lr_config.lr_embedding,       'name': 'graph_embedding'},
            {'params': ppo.policy.job_actor.parameters(),            'lr': lr_config.lr_job_actor,       'name': 'job_actor'},
        ]
                       
        self.logger.section("[Optimizer]")
        self.logger.subsection(f"Operator Actor  = {lr_config.lr_operator_actor}")
        self.logger.subsection(f"Vehicle Actor   = {lr_config.lr_vehicle_actor}")
        self.logger.subsection(f"Critic          = {lr_config.lr_critic}")
        self.logger.subsection(f"Graph Embedding = {lr_config.lr_embedding}")
        self.logger.subsection(f"Job Actor       = {lr_config.lr_job_actor} \n")

        optimizer = optim.Adam(param_groups, eps=1e-5)
        ppo.optimizer = optimizer
        
        ppo.lr_scheduler = LRScheduler(
            optimizer    = optimizer,
            warmup_steps = lr_config.lr_warmup_steps,
            decay_steps  = lr_config.lr_decay_steps,
            lr_min       = lr_config.lr_min,
        )

        ppo.entropy_scheduler = EntropyScheduler(
            start_coef   = entropy_config.entropy_start,
            end_coef     = entropy_config.entropy_end,
            anneal_steps = entropy_config.entropy_anneal_steps,
            warmup_steps = lr_config.lr_warmup_steps,
        )

        ppo.early_stopping = EpochEarlyStopping(self.config.ppo.kl_divergence_threshold)

        self.logger.section("[Schedulers]")
        self.logger.subsection(f"LR Warmup Steps  = {lr_config.lr_warmup_steps}")
        self.logger.subsection(f"LR Decay Steps   = {lr_config.lr_decay_steps}")
        self.logger.subsection(f"LR Min           = {lr_config.lr_min}")
        self.logger.subsection(f"Entropy Schedule = {entropy_config.entropy_start} -> {entropy_config.entropy_end} over {entropy_config.entropy_anneal_steps} steps")
        self.logger.subsection(f"KL Threshold     = {self.config.ppo.kl_divergence_threshold} \n")

        ppo.current_entropy_coef = entropy_config.entropy_start
        
        if self.load_checkpoint:
            self.checkpoint.load(ppo, self, self.dataset)
               
        self.logger.subsection("PPO initialization complete \n")
        return ppo
    
    def state(self):
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
        self.ppo.update()
        
    def run_episode(self, dataset_item):
        experiences = []
        max_steps = self.config.training.max_steps_per_episode
        
        self.environment.load_from_dataset(dataset_item)
        for step_in_episode in range(max_steps):
            
            if step_in_episode == 0:
                graph     = dataset_item["graph"]
                mask_info = dataset_item["mask_info"]
            else:
                graph, mask_info = self.environment.observe()

            with torch.no_grad():
                if self.attached:
                    graph      = graph.to(self.device)
                    ppo_output = self.ppo.policy.select_action(graph, mask_info)
                    self.tensor_logger.save_markdown(path=os.path.join(self.config.io.logdir, "tensor_shape.md"), title=f"Tensor Shapes")
                    self.tensor_logger.detach()
                    self.attached = False
                    self.logger.section("[Tensor Logger]")
                    self.logger.subsection("Tensor Shape Saved - Detaching Tensor Logger \n")
                else:
                    graph      = graph.to(self.device)
                    ppo_output = self.ppo.policy.select_action(graph, mask_info)

            action         = ppo_output["action"]
            value          = ppo_output["state_value"].item()
            operator_index = action.operator

            old_state, next_state = self.environment.apply_action(action)

            rewards, costs = self.environment.step(old_state, next_state, operator_index)
            reward = sum(rewards.values())

            self.operator_stats['count'][operator_index] += 1
            self.operator_stats['rewards'][operator_index].append(reward)

            self.telemetry.step(rewards, costs, value, self.global_step_counter)

            experience = {
                "graph"               : graph,
                "mask_info"           : mask_info,
                "reward"              : float(reward),
                "action"              : ppo_output["action"],
                "log_prob_operator"   : ppo_output["log_prob_operator"].detach().cpu(),
                "log_prob_vehicle"    : ppo_output["log_prob_vehicle"].detach().cpu(),
                "log_prob_job"        : ppo_output["log_prob_job"].detach().cpu(),
                "state_value"         : ppo_output["state_value"].detach().cpu(),
                "old_operator_logits" : ppo_output["old_operator_logits"].detach().cpu(),
                "old_vehicle_logits"  : ppo_output["old_vehicle_logits"].detach().cpu(),
                "old_job_logits"      : ppo_output["old_job_logits"].detach().cpu(),
                "bootstrap_value"     : 0.0,
                "done"                : False,
            }

            experiences.append(experience)
            self.environment.current_state = next_state

        if experiences:
            experiences[-1]["done"] = True

            tail_graph, _ = self.environment.observe()
            with torch.no_grad():
                _, _, _, tail_value = self.ppo.policy(tail_graph.to(self.device))
            experiences[-1]["bootstrap_value"] = float(tail_value.item())

        return experiences

    def run_chunk(self, chunk_data):
        self.logger.section("[Chunk Processing]")
        self.logger.subsection(f"Total episodes in chunk : {len(chunk_data)}")
        self.logger.subsection(f"Starting episode index  : {self.episode_index} \n")
        
        episodes_processed = 0
        for dataset_item in tqdm(chunk_data, desc="Processing episodes in chunk", leave=False):
            experiences = self.run_episode(dataset_item)
            
            episode_reward = sum([exp['reward'] for exp in experiences])
            episode_length = len(experiences)
            
            for experience in experiences:
                self.ppo.memory.add(**experience)
                self.global_step_counter += 1

            # Note: done flag already set in run_episode for last experience
            self.episode_index += 1
            episodes_processed += 1
            
            self.telemetry.episode(episode_reward, episode_length, self.operator_stats, self.episode_index)
            self.operator_stats = {'count': {i: 0 for i in range(4)}, 'rewards': {i: [] for i in range(4)}}

        self.logger.subsection(f"Chunk complete: {episodes_processed} episodes processed")
        return episodes_processed
     
    def train(self):
        self.ppo.train()
        self.logger.section("[Training Start]")
        self.logger.subsection(f"Device Name: {torch.cuda.get_device_name(0)}")
        self.logger.subsection(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        self.logger.subsection(f"CUDA Version: {torch.version.cuda} \n")

        self.logger.section("[Output Directory]")
        self.logger.subsection(f"Log Directory: {self.config.io.logdir} \n")
        
        self.logger.section("Model Summary")
        self.logger.subsection("Generating model architecture summary")
        self.summary.run()
        self.summary.save_markdown(os.path.join(self.config.io.logdir, "model_summary.md"), title="PPO Model Summary")
        self.logger.subsection("Model summary saved \n")
        
        self.logger.section("[Training Loop]")
        chunk_paths = self.dataset.chunk_paths
        total_chunks = len(chunk_paths)
        self.logger.subsection(f"Total chunks to process: {total_chunks} \n")

        for chunk_idx, chunk_path in tqdm(enumerate(chunk_paths), desc="Training chunks", total=total_chunks):
            chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)

            episodes_processed = self.run_chunk(chunk_data)

            self.telemetry.episodes_processed(episodes_processed, chunk_idx)

            self.ppo_update()
            self.checkpoint.save(self.ppo, self)

            progress_pct = ((chunk_idx + 1) / total_chunks) * 100
            self.logger.subsection(f"Overall Progress: {progress_pct:.1f}% ({chunk_idx + 1}/{total_chunks} chunks)")
            self.telemetry.chunk_progress(chunk_idx, total_chunks)
            
            del chunk_data
            gc.collect()
            # Removed torch.cuda.empty_cache() - causes re-allocation overhead
        
        self.logger.section("[Training Complete]")
        self.logger.subsection("Training loop finished")
 
        self.ppo.memory.clear()
        self.logger.subsection("Training finished successfully")
        return self.ppo