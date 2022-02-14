# -*- coding: utf-8 -*-
"""Common 2D convolutions
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from typing import Tuple, List

from splearn.nn.modules.functional import Swish
from splearn.nn.utils import get_class_name


class Conv2d(nn.Module):
    """
    Input: 4-dim tensor
        Shape [batch, in_channels, H, W]
    Return: 4-dim tensor
        Shape [batch, out_channels, H, W]
        
    Args:
        in_channels : int
            Should match input `channel`
        out_channels : int
            Return tensor with `out_channels`
        kernel_size : int or 2-dim tuple
        stride : int or 2-dim tuple, default: 1
        padding : int or 2-dim tuple or True
            Apply `padding` if given int or 2-dim tuple. Perform TensorFlow-like 'SAME' padding if True
        dilation : int or 2-dim tuple, default: 1
        groups : int or 2-dim tuple, default: 1
        w_in: int, optional
            The size of `W` axis. If given, `w_out` is available.
    
    Usage:
        x = torch.randn(1, 22, 1, 256)
        conv1 = Conv2dSamePadding(22, 64, kernel_size=17, padding=True, w_in=256)
        y = conv1(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="SAME", dilation=1, groups=1, w_in=None, bias=True):
        super().__init__()
        
        padding = padding
        self.kernel_size = kernel_size = kernel_size
        self.stride = stride = stride
        self.dilation = dilation = dilation
        
        self.padding_same = False
        if padding == "SAME":
            self.padding_same = True
            padding = (0,0)
        
        if isinstance(padding, int):
            padding = (padding, padding)
            
        if isinstance(kernel_size, int):
            self.kernel_size = kernel_size = (kernel_size, kernel_size)
            
        if isinstance(stride, int):
            self.stride = stride = (stride, stride)
        
        if isinstance(dilation, int):
            self.dilation = dilation = (dilation, dilation)
            
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0 if padding==True else padding, 
            dilation=dilation, 
            groups=groups,
            bias=bias
        )
        
        if w_in is not None:
            self.w_out = int( ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1]-1)-1) / 1) + 1 )
        if self.padding_same == "SAME": # if SAME, then replace, w_out = w_in, obviously
            self.w_out = w_in
            
    def forward(self, x):
        if self.padding_same == True:
            x = self.pad_same(x, self.kernel_size, self.stride, self.dilation)
        return self.conv(x)
    
    # Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
    def get_same_padding(self, x: int, k: int, s: int, d: int):
        return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

    # Dynamically pad input x with 'SAME' padding for conv with specified args
    def pad_same(self, x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
        ih, iw = x.size()[-2:]
        pad_h, pad_w = self.get_same_padding(ih, k[0], s[0], d[0]), self.get_same_padding(iw, k[1], s[1], d[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
        return x


class Conv2dBlockELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, activation=nn.ELU, w_in=None):
        super(Conv2dBlockELU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )

        if w_in is not None:
            self.w_out = int( ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1]-1)-1) / 1) + 1 )

    def forward(self, x):
        return self.conv(x)


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth=1, padding=0, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
            
        if isinstance(kernel_size, tuple):
            padding = (
                kernel_size[0]//2 if kernel_size[0]-1 != 0 else 0,
                kernel_size[1]//2 if kernel_size[1]-1 != 0 else 0
            )
            
        self.depthwise = DepthwiseConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


####

class Conv2dExtractor(nn.Module):
    r"""
    Provides inteface of convolutional extractor.
    Note:
        Do not use this class directly, use one of the sub classes.
        Define the 'self.conv' class variable.
    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths
    Returns: outputs, output_lengths
        - **outputs**: Tensor produced by the convolution
        - **output_lengths**: Tensor containing sequence lengths produced by the convolution
    """
    supported_activations = {
        'hardtanh': nn.Hardtanh(0, 20, inplace=True),
        'relu': nn.ReLU(inplace=True),
        'elu': nn.ELU(inplace=True),
        'leaky_relu': nn.LeakyReLU(inplace=True),
        'gelu': nn.GELU(),
        'swish': Swish(),
    }

    def __init__(self, input_dim: int, activation: str = 'hardtanh') -> None:
        super(Conv2dExtractor, self).__init__()
        self.input_dim = input_dim
        self.activation = Conv2dExtractor.supported_activations[activation]
        self.conv = None

    def get_output_lengths(self, seq_lengths: torch.Tensor):
        assert self.conv is not None, "self.conv should be defined"

        for module in self.conv:
            if isinstance(module, nn.Conv2d):
                numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
                seq_lengths = numerator.float() / float(module.stride[1])
                seq_lengths = seq_lengths.int() + 1

            elif isinstance(module, nn.MaxPool2d):
                seq_lengths >>= 1

        return seq_lengths.int()

    def get_output_dim(self):
        if get_class_name(self) == "VGGExtractor":
            output_dim = (self.input_dim - 1) << 5 if self.input_dim % 2 else self.input_dim << 5

        elif get_class_name(self) == "DeepSpeech2Extractor":
            output_dim = int(math.floor(self.input_dim + 2 * 20 - 41) / 2 + 1)
            output_dim = int(math.floor(output_dim + 2 * 10 - 21) / 2 + 1)
            output_dim <<= 5

        elif get_class_name(self) == "Conv2dSubsampling":
            factor = ((self.input_dim - 1) // 2 - 1) // 2
            output_dim = self.out_channels * factor

        else:
            raise ValueError(f"Unsupported Extractor : {self.extractor}")

        return output_dim

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        inputs: torch.FloatTensor (batch, time, dimension)
        input_lengths: torch.IntTensor (batch)
        """
        outputs, output_lengths = self.conv(inputs.unsqueeze(1).transpose(2, 3), input_lengths)

        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.view(batch_size, seq_lengths, channels * dimension)

        return outputs, output_lengths

class Conv2dSubsampling(Conv2dExtractor):
    r"""
    Convolutional 2D subsampling (to 1/4 length)
    Args:
        input_dim (int): Dimension of input vector
        in_channels (int): Number of channels in the input vector
        out_channels (int): Number of channels produced by the convolution
        activation (str): Activation function
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs
        - **input_lengths** (batch): list of sequence input lengths
    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(
            self,
            input_dim: int,
            in_channels: int,
            out_channels: int,
            activation: str = 'relu',
    ) -> None:
        super(Conv2dSubsampling, self).__init__(input_dim, activation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = MaskConv2d(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
                self.activation,
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
                self.activation,
            )
        )

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, input_lengths = super().forward(inputs, input_lengths)
        output_lengths = input_lengths >> 2
        output_lengths -= 1
        return outputs, output_lengths
    
class MaskConv2d(nn.Module):
    r"""
    Masking Convolutional Neural Network
    Adds padding to the output of the module based on the given lengths.
    This is to ensure that the results of the model do not change when batch sizes change during inference.
    Input needs to be in the shape of (batch_size, channel, hidden_dim, seq_len)
    Refer to https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
    Copyright (c) 2017 Sean Naren
    MIT License
    Args:
        sequential (torch.nn): sequential list of convolution layer
    Inputs: inputs, seq_lengths
        - **inputs** (torch.FloatTensor): The input of size BxCxHxT
        - **seq_lengths** (torch.IntTensor): The actual length of each sequence in the batch
    Returns: output, seq_lengths
        - **output**: Masked output from the sequential
        - **seq_lengths**: Sequence length of output from the sequential
    """
    def __init__(self, sequential: nn.Sequential) -> None:
        super(MaskConv2d, self).__init__()
        self.sequential = sequential

    def forward(self, inputs: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        output = None

        for module in self.sequential:
            output = module(inputs)
            mask = torch.BoolTensor(output.size()).fill_(0)

            if output.is_cuda:
                mask = mask.cuda()

            seq_lengths = self._get_sequence_lengths(module, seq_lengths)

            for idx, length in enumerate(seq_lengths):
                length = length.item()

                if (mask[idx].size(2) - length) > 0:
                    mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lengths

    def _get_sequence_lengths(self, module: nn.Module, seq_lengths: Tensor) -> Tensor:
        r"""
        Calculate convolutional neural network receptive formula
        Args:
            module (torch.nn.Module): module of CNN
            seq_lengths (torch.IntTensor): The actual length of each sequence in the batch
        Returns: seq_lengths
            - **seq_lengths**: Sequence length of output from the module
        """
        if isinstance(module, nn.Conv2d):
            numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lengths = numerator.float() / float(module.stride[1])
            seq_lengths = seq_lengths.int() + 1

        elif isinstance(module, nn.MaxPool2d):
            seq_lengths >>= 1

        return seq_lengths.int()