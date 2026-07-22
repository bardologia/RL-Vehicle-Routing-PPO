import math


class LRScheduler:
    def __init__(self, optimizer, warmup_steps=1000, decay_steps=100000, lr_min=1e-5):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.decay_steps  = decay_steps
        self.lr_min       = lr_min
        self.current_step = 0

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

        return self.optimizer.param_groups[0]['lr']

    def get_lr(self):
        return {f'group_{i}': pg['lr'] for i, pg in enumerate(self.optimizer.param_groups)}

    def set_step(self, step):
        self.current_step = step


class EntropyScheduler:
    def __init__(self, start_coef=0.02, end_coef=0.001, anneal_steps=50000, warmup_steps=0):
        self.start_coef   = start_coef
        self.end_coef     = end_coef
        self.anneal_steps = anneal_steps
        self.warmup_steps = warmup_steps

        self.current_step = 0

    def step(self):
        self.current_step += 1
        return self.get_coef()

    def get_coef(self):
        return self._linear_anneal()

    def _linear_anneal(self):
        if self.current_step <= self.warmup_steps:
            return self.start_coef

        adjusted_step = self.current_step - self.warmup_steps
        progress = min(1.0, adjusted_step / self.anneal_steps)
        return self.start_coef + (self.end_coef - self.start_coef) * progress

    def set_step(self, step):
        self.current_step = step


class EpochEarlyStopping:
    def __init__(self, threshold):
        self.threshold = threshold

    def should_stop(self, kl_divergence):
        return kl_divergence > self.threshold
