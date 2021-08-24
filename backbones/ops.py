""" Operations """
import torch
import torch.nn as nn
from backbones import genotypes as gt
import torch.nn.functional as F

# ------------------------------ own stuff ----------------------------------

# pool 3x3, 5x5 max, avg, conv 3x3, 5x5, 7x7 dw

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: \
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'dw_conv_3x3': lambda C, stride, affine: DWConv(C, C, 3, stride, 1, affine=affine),
    'dw_conv_5x5': lambda C, stride, affine: DWConv(C, C, 5, stride, 2, affine=affine),
    'dw_conv_7x7': lambda C, stride, affine: DWConv(C, C, 7, stride, 3, affine=affine),
    'dw_conv_1x1': lambda C, stride, affine: DWConv(C, C, 1, stride, 0, affine=affine),
}


class StdConv(nn.Module):
    """ Standard conv
    PReLU - DWConv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.PReLU(C_in),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

class DWConv(nn.Module):
    """ Depthwise separable conv
    ReLU - Depthwise separable - Pointwise - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.PReLU(C_in),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

class con1x1(nn.Module):
    """ Depthwise separable conv
    ReLU - Depthwise separable - Pointwise - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.PReLU(C_in),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in,
                      bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.PReLU(C_in)

        self.conv11 = nn.Conv2d(C_in, C_in, 1, 2, 0, groups=C_in,
                      bias=False)
        self.conv12 = nn.Conv2d(C_in, C_out // 2, 1, stride=1, padding=0, bias=False)

        self.conv21 = nn.Conv2d(C_in, C_in, 1, 2, 0, groups=C_in,
                      bias=False)
        self.conv22 = nn.Conv2d(C_in, C_out // 2, 1, stride=1, padding=0, bias=False)


        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        print("call Factorized")
        x = self.relu(x)
        first = self.conv11(x)
        first = self.conv12(first)

        second = self.conv21(x[:, :, 1:, 1:])
        second = self.conv22(second)

        out = torch.cat([first, second],dim=1)
        out = self.bn(out)
        return out

class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))
