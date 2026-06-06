from pytorch_lightning.callbacks import Callback


class BatchLearningRateDecay(Callback):
    def __init__(self, initial_lr, final_lr, total_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_steps = total_steps

        self.steps = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Calculate the new learning rate
        current_step = trainer.global_step

        if self.total_steps < current_step:
            lr = self.final_lr
        else:
            lr = self.initial_lr + (self.final_lr - self.initial_lr) * (current_step / self.total_steps)

        # Update the learning rate in the optimizer
        for param_group in trainer.optimizers[0].param_groups:
            param_group['lr'] = lr