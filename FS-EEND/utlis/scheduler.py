from torch.optim.lr_scheduler import _LRScheduler

class NoamScheduler(_LRScheduler):
    """
    See https://arxiv.org/pdf/1706.03762.pdf
    lrate = d_model**(-0.5) * \
            min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
    Args:
        d_model: int
            The number of expected features in the encoder inputs.
        warmup_steps: int
            The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, d_model, warmup_steps, scale, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale
        super(NoamScheduler, self).__init__(optimizer, last_epoch)
        print("Noam initializing...")
        if self.last_epoch == -1:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
            self.last_epoch = 0
    
    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.scale * self.d_model ** (-0.5) * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]