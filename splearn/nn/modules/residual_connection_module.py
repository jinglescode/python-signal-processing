import torch.nn as nn
from torch import Tensor
from typing import Optional


class ResidualConnectionModule(nn.Module):
    r"""
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(
            self,
            module: nn.Module,
            module_factor: float = 1.0,
            input_factor: float = 1.0,
    ) -> None:
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)
        else:
            return (self.module(inputs, mask) * self.module_factor) + (inputs * self.input_factor)
