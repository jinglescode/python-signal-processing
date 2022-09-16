import torch
from torch import nn
from .Multitask import MultitaskSSVEP

class MultitaskSSVEPClassifier(nn.Module):
    """
    Using multi-task learning to capture signals simultaneously from the fovea efficiently and the neighboring targets in the peripheral vision generate a visual response map. A calibration-free user-independent solution, desirable for clinical diagnostics. A stepping stone for an objective assessment of glaucoma patients’ visual field.
    Learn more about this model at https://jinglescode.github.io/ssvep-multi-task-learning/
    This model is a multi-class classifier.
    Usage:
        model = MultitaskSSVEPClassifier(
            num_channel=11,
            num_classes=40,
            signal_length=250,
        )
        x = torch.randn(2, 11, 250)
        print("Input shape:", x.shape) # torch.Size([2, 11, 250])
        y = model(x)
        print("Output shape:", y.shape) # torch.Size([2, 40])
        
    Cite:
        @inproceedings{khok2020deep,
          title={Deep Multi-Task Learning for SSVEP Detection and Visual Response Mapping},
          author={Khok, Hong Jing and Koh, Victor Teck Chang and Guan, Cuntai},
          booktitle={2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
          pages={1280--1285},
          year={2020},
          organization={IEEE}
        }
    """
    
    def __init__(self, num_channel=10, num_classes=4, signal_length=1000, filters_n1=4, kernel_window_ssvep=59, kernel_window=19, conv_3_dilation=4, conv_4_dilation=4):
        super().__init__()
        self.base = MultitaskSSVEP(num_channel, num_classes, signal_length, filters_n1, kernel_window_ssvep, kernel_window, conv_3_dilation, conv_4_dilation)
        self.fc = nn.Linear(num_classes*2, out_features=num_classes)

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def test():
    model = MultitaskSSVEPClassifier(
        num_channel=11,
        num_classes=40,
        signal_length=250,
    )

    x = torch.randn(2, 11, 250)
    print("Input shape:", x.shape) # torch.Size([2, 11, 250])
    y = model(x)
    print("Output shape:", y.shape) # torch.Size([2, 40])

if __name__ == "__main__":
    test()