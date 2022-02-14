from splearn.nn.base import LightningModelClassifier
from splearn.nn.models import SimSiam
from splearn.nn.utils import get_backbone_and_fc
from splearn.nn.loss import LabelSmoothCrossEntropyLoss


class SSLClassifier(LightningModelClassifier):
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
        projection_size = kwargs["projection_size"] if "projection_size" in kwargs else 2048
        num_proj_mlp_layers = kwargs["num_proj_mlp_layers"] if "num_proj_mlp_layers" in kwargs else 3
        
        backbone, classifier = get_backbone_and_fc(model)
        self.ssl_network = SimSiam(backbone=backbone, projection_size=projection_size, num_proj_mlp_layers=num_proj_mlp_layers)
        self.classifier_network = classifier

    def forward(self, x):
        features = self.ssl_network.backbone(x)
        y_hat = self.classifier_network(features)
        return y_hat
    
    def train_val_step(self, batch, batch_idx):
        x1, x2, y = batch
        
        out = self.ssl_network(x1, x2)
        loss_recon = out['loss']
        features = out['features']
        
        y_hat = self.classifier_network(features)
        loss_cross_entropy = self.criterion_classifier(y_hat, y.long()) # self.criterion_classifier(y_hat, y.long()) # F.cross_entropy(y_hat, y.long()) 
        
        loss = loss_recon + loss_cross_entropy
        
        return y_hat, y, loss, loss_recon, loss_cross_entropy

    def training_step(self, batch, batch_idx):
        y_hat, y, loss, loss_recon, loss_cross_entropy = self.train_val_step(batch, batch_idx)
        acc = self.train_acc(y_hat, y.long())
        self.log('train_loss', loss, on_step=True)
        self.log('train_loss_recon', loss_recon, on_step=True)
        self.log('train_loss_cross_entropy', loss_cross_entropy, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss, loss_recon, loss_cross_entropy = self.train_val_step(batch, batch_idx)
        acc = self.valid_acc(y_hat, y.long())
        self.log('valid_loss', loss, on_step=True)
        self.log('valid_loss_recon', loss_recon, on_step=True)
        self.log('valid_loss_cross_entropy', loss_cross_entropy, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion_classifier(y_hat, y.long()) # self.criterion_classifier(y_hat, y.long()) # F.cross_entropy(y_hat, y.long()) 
        acc = self.test_acc(y_hat, y.long())
        self.log('test_loss', loss)
        return loss
