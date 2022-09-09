import torchmetrics
from splearn.nn.base import LightningModel
from splearn.nn.loss import LabelSmoothCrossEntropyLoss


class LightningModelClassifier(LightningModel):
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
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        
        self.criterion_classifier = LabelSmoothCrossEntropyLoss(smoothing=0.3) # F.cross_entropy()
    
    def build_model(self, model):
        self.model = model

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat
    
    def step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion_classifier(y_hat, y.long()) # self.criterion_classifier(y_hat, y.long()) # F.cross_entropy(y_hat, y.long())
        return y_hat, y, loss

    def training_step(self, batch, batch_idx):
        y_hat, y, loss = self.step(batch, batch_idx)
        acc = self.train_acc(y_hat, y.long())
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss = self.step(batch, batch_idx)
        acc = self.valid_acc(y_hat, y.long())
        self.log('valid_loss', loss, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y, loss = self.step(batch, batch_idx)
        acc = self.test_acc(y_hat, y.long())
        self.log('test_loss', loss)
        return loss
    
    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_acc.compute())
        
    def validation_epoch_end(self, outs):
        self.log('valid_acc_epoch', self.valid_acc.compute())
    
    def test_epoch_end(self, outs):
        self.log('test_acc_epoch', self.test_acc.compute())