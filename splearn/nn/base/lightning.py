from pytorch_lightning import LightningModule
from splearn.nn.optimization import get_scheduler, get_optimizer, get_num_steps


class LightningModel(LightningModule):
    def __init__(
        self
    ):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def configure_optimizers(self):

        optimizer = get_optimizer(
            name=self.hparams.optimizer,
            model=self,
            lr=self.hparams.optimizer_learning_rate,
            weight_decay=self.hparams.optimizer_weight_decay,
            epsilon=self.hparams.optimizer_epsilon
        )
        
        total_train_steps, num_warmup_steps = get_num_steps(self)
        
        scheduler = get_scheduler(
            name=self.hparams.scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_train_steps,
        )
        
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]
