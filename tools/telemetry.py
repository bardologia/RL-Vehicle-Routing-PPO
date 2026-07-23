import torch
import torch.nn as nn


class PPOTelemetry:
    def __init__(self, tracker, config):
        self.tracker = tracker
        self.config  = config
        self.gates   = config.telemetry

    def step(self, rewards, costs, state_value, step):
        if not self.tracker.active or step % self.gates.step_every != 0:
            return

        self.tracker.log_metrics('step/reward', rewards, step)
        self.tracker.log_metrics('step/cost', costs, step)
        self.tracker.log_scalar('step/state_value', state_value, step)

    def episode(self, reward, length, operator_stats, episode_index):
        if not self.tracker.active or episode_index % self.gates.episode_every != 0:
            return

        self.tracker.log_scalar('episode/total_reward', reward, episode_index)
        self.tracker.log_scalar('episode/length', length, episode_index)

        total = sum(operator_stats['count'].values())
        if total == 0:
            return

        frequency   = {f'op{i}': operator_stats['count'][i] / total for i in operator_stats['count']}
        avg_rewards = {
            f'op{i}': (sum(rewards) / len(rewards) if rewards else 0.0)
            for i, rewards in operator_stats['rewards'].items()
        }

        self.tracker.log_metrics('episode/operator_frequency', frequency, episode_index)
        self.tracker.log_metrics('episode/operator_avg_reward', avg_rewards, episode_index)

    def episodes_processed(self, count, chunk_index):
        self.tracker.log_scalar('batch/episodes_processed', count, chunk_index)

    def chunk_progress(self, chunk_index, total_chunks):
        self.tracker.log_scalar('batch/chunk_progress', chunk_index / total_chunks, chunk_index)

    def buffer_size(self, size, update_step):
        self.tracker.log_scalar('batch/buffer_size', size, update_step)

    def baseline(self, batch_data, update_step):
        if not self.tracker.active:
            return

        values  = batch_data["values"]
        returns = batch_data["returns"]

        explained_variance = 0.0
        if returns.std() > 1e-8:
            explained_variance = float(1.0 - ((returns - values).var() / (returns.var() + 1e-8)))

        self.tracker.log_metrics('batch/baseline_stats', {
            "advantage_mean"     : batch_data["advantages"].mean(),
            "advantage_std"      : batch_data["advantages"].std(),
            "return_mean"        : returns.mean(),
            "return_std"         : returns.std(),
            "reward_mean"        : batch_data["rewards"].mean(),
            "reward_std"         : batch_data["rewards"].std(),
            "value_mean"         : values.mean(),
            "value_std"          : values.std(),
            "explained_variance" : explained_variance,
        }, update_step)

    def sample(self, sample_step, advantage, target_return, old_log_probs, new_log_probs, distributions, policy_loss_dict, value_loss_dict, entropy_dict, kl_dict, entropy_loss, total_loss):
        if not self.tracker.active or sample_step % self.gates.sample_every != 0:
            return

        prob_ratios = self._prob_ratios(old_log_probs, new_log_probs)

        self.tracker.log_metrics('sample/loss', self._component_losses(advantage, prob_ratios), sample_step)
        self.tracker.log_metrics('sample/clip_fraction', self._clip_fractions(prob_ratios), sample_step)
        self.tracker.log_metrics('sample/prob_ratio', prob_ratios, sample_step)
        self.tracker.log_metrics('sample/old_log_probs', old_log_probs, sample_step)
        self.tracker.log_metrics('sample/new_log_probs', new_log_probs, sample_step)
        self.tracker.log_metrics('sample/entropy', entropy_dict, sample_step)
        self.tracker.log_metrics('sample/kl_divergence', kl_dict, sample_step)
        self.tracker.log_metrics('sample/policy_loss', policy_loss_dict, sample_step)
        self.tracker.log_metrics('sample/value_loss', value_loss_dict, sample_step)
        self.tracker.log_scalar('sample/entropy_loss', entropy_loss, sample_step)
        self.tracker.log_scalar('sample/total_loss', total_loss, sample_step)
        self.tracker.log_scalar('sample/advantage', advantage, sample_step)
        self.tracker.log_scalar('sample/target_return', target_return, sample_step)

        for head, distribution in distributions.items():
            probs = distribution.probs.detach()
            self.tracker.log_metrics(f'sample/{head}_distribution', {
                'max_prob'  : probs.max(),
                'mean_prob' : probs.mean(),
                'min_prob'  : probs.min(),
            }, sample_step)

    def batch(self, mean_loss, mean_kl, batch_step):
        self.tracker.log_scalar('batch/mean_loss', mean_loss, batch_step)
        self.tracker.log_scalar('batch/mean_kl', mean_kl, batch_step)

    def gradients(self, modules_dict, batch_step):
        if not self.tracker.active:
            return

        grad_norms = {}
        grad_stats = {}

        for module_name, module in modules_dict.items():
            grad_norms[module_name] = nn.utils.clip_grad_norm_(module.parameters(), max_norm=float('inf')).item()
            grad_stats.update(self._gradient_stats(module, module_name))

            if batch_step % self.gates.layer_gradients_every == 0:
                self._layer_gradients(module_name, module, batch_step)

        self.tracker.log_metrics('batch/gradients_norms', grad_norms, batch_step)
        self.tracker.log_metrics('batch/gradients_stats', grad_stats, batch_step)

    def grad_norm(self, total_norm, batch_step):
        self.tracker.log_scalar('batch/grad_norm_before_clip', total_norm, batch_step)

    def epoch(self, mean_loss, mean_kl, epoch_step):
        self.tracker.log_scalar('epoch/mean_loss', mean_loss, epoch_step)
        self.tracker.log_scalar('epoch/mean_kl', mean_kl, epoch_step)

    def early_stop(self, epoch, epoch_step):
        self.tracker.log_scalar('batch/epoch_early_stop_epoch', epoch, epoch_step)

    def learning_rates(self, optimizer, step):
        if not self.tracker.active:
            return

        rates = {group['name']: group['lr'] for group in optimizer.param_groups}
        self.tracker.log_metrics('batch/learning_rate', rates, step)

    def entropy_coefficient(self, value, step):
        self.tracker.log_scalar('batch/entropy_coefficient', value, step)

    def pretrain_rollout(self, episode_reward, operator_counts, episode_index):
        if not self.tracker.active or episode_index % self.gates.episode_every != 0:
            return

        self.tracker.log_scalar('pretrain/rollout_reward', episode_reward, episode_index)

        total = sum(operator_counts.values())
        if total == 0:
            return

        frequency = {f'op{operator}': count / total for operator, count in operator_counts.items()}
        self.tracker.log_metrics('pretrain/operator_frequency', frequency, episode_index)

    def pretrain_batch(self, loss_values, batch_step):
        if not self.tracker.active or batch_step % self.gates.step_every != 0:
            return

        self.tracker.log_metrics('pretrain/loss', loss_values, batch_step)

    def pretrain_epoch(self, mean_loss, accuracy, epoch_step):
        self.tracker.log_scalar('pretrain/epoch_loss', mean_loss, epoch_step)
        self.tracker.log_metrics('pretrain/accuracy', accuracy, epoch_step)

    def _prob_ratios(self, old_log_probs, new_log_probs):
        return {
            "operator" : torch.exp(new_log_probs["op"] - old_log_probs["op"]),
            "vehicle"  : torch.exp(new_log_probs["veh"] - old_log_probs["veh"]),
            "job"      : torch.exp(new_log_probs["job"] - old_log_probs["job"]),
        }

    def _component_losses(self, advantage, prob_ratios):
        clip = self.config.ppo.clip_ratio

        losses = {}
        for head, ratio in prob_ratios.items():
            clipped      = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
            losses[head] = -torch.min(ratio * advantage, clipped * advantage)

        return losses

    def _clip_fractions(self, prob_ratios):
        clip = self.config.ppo.clip_ratio
        return {head: (torch.abs(ratio - 1.0) > clip).float().mean() for head, ratio in prob_ratios.items()}

    def _gradient_stats(self, module, module_name):
        grad_values  = []
        param_values = []

        for param in module.parameters():
            if param.grad is not None:
                grad_values.append(param.grad.detach().view(-1))
                param_values.append(param.detach().view(-1))

        if not grad_values:
            return {}

        all_grads  = torch.cat(grad_values)
        all_params = torch.cat(param_values)

        stats = {
            f'{module_name}/grad_mean'     : all_grads.mean(),
            f'{module_name}/grad_std'      : all_grads.std(unbiased=False),
            f'{module_name}/grad_max'      : all_grads.max(),
            f'{module_name}/grad_min'      : all_grads.min(),
            f'{module_name}/grad_abs_mean' : all_grads.abs().mean(),
            f'{module_name}/has_nan'       : torch.isnan(all_grads).any(),
            f'{module_name}/has_inf'       : torch.isinf(all_grads).any(),
        }

        param_norm = all_params.norm().item()
        if param_norm > 1e-8:
            stats[f'{module_name}/grad_param_ratio'] = all_grads.norm().item() / param_norm

        return stats

    def _layer_gradients(self, module_name, module, batch_step):
        layer_stats = {}

        for name, param in module.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                layer_stats[f'{module_name}/{name}/norm'] = grad.norm()
                layer_stats[f'{module_name}/{name}/mean'] = grad.mean()
                layer_stats[f'{module_name}/{name}/std']  = grad.std(unbiased=False)

        if layer_stats:
            self.tracker.log_metrics(f'batch/gradients_layers/{module_name}', layer_stats, batch_step)
