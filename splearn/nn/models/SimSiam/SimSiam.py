import torch
import torch.nn as nn
import torch.nn.functional as F 


class SimSiam(nn.Module):
    def __init__(self, backbone, projection_size=2048, hidden_dim=None, num_proj_mlp_layers=3):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = int(projection_size/4)
        
        self.backbone = backbone
        
        self.projector = projection_MLP(
            in_dim=self.backbone.output_dim, 
            hidden_dim=projection_size, 
            out_dim=projection_size,
            num_layers=num_proj_mlp_layers,
        )
        
        self.predictor = prediction_MLP(
            in_dim=projection_size, 
            hidden_dim=hidden_dim, 
            out_dim=projection_size
        )
                    
    def forward(self, x1, x2):
        
        latent = self.backbone(x1)
        z1 = self.projector(latent)
        p1 = self.predictor(z1)
        
        z2 = self.backbone(x2)
        z2 = self.projector(z2)
        p2 = self.predictor(z2)
        
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L, 'features': latent}
    

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    
    elif version == 'mse':
        return F.mse_loss(p, z.detach(), reduction='mean')
    
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, num_layers=3):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN applied to each fully-connected (fc) layer, including its output fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = num_layers
        
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied to its hidden fc layers. Its output fc does not have BN (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, and h’s hidden layer’s dimension is 512, making h a bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 



