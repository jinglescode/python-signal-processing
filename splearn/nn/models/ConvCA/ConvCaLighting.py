from splearn.nn.base import LightningModelClassifier
from splearn.nn.models import ConvCA
from splearn.nn.utils import get_backbone_and_fc
from splearn.nn.loss import LabelSmoothCrossEntropyLoss


class ConvCaLighting(LightningModelClassifier):
    def __init__(
        self,
        optimizer="adamw",
        scheduler="cosine_with_warmup",
        optimizer_learning_rate: float=1e-3,
        optimizer_epsilon: float=1e-6,
        optimizer_weight_decay: float=0.0005,
        scheduler_warmup_epochs: int=10,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.criterion_classifier = LabelSmoothCrossEntropyLoss(smoothing=0.3)
    
    def build_model(self, model, **kwargs):
        self.model = model

    def forward(self, x, ref):
        y_hat = self.model(x, ref)
        return y_hat
    
    def train_val_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat = self.forward(x1, x2)
        loss = self.criterion_classifier(y_hat, y.long())
        return y_hat, y, loss

    def training_step(self, batch, batch_idx):
        y_hat, y, loss = self.train_val_step(batch, batch_idx)
        acc = self.train_acc(y_hat, y.long())
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss = self.train_val_step(batch, batch_idx)
        acc = self.valid_acc(y_hat, y.long())
        self.log('valid_loss', loss, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y, loss = self.train_val_step(batch, batch_idx)
        acc = self.test_acc(y_hat, y.long())
        self.log('test_loss', loss)
        return loss
