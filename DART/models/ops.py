""" Operations """
import torch
import torch.nn as nn
from models import genotypes as gt

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
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN
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

# ---------------------------------------

# OPS = {
#     'none': lambda C, stride, affine: Zero(stride),
#     'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
#     'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
#     'skip_connect': lambda C, stride, affine: \
#         Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#     'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
#     'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
#     'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
#     'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine), # 5x5
#     'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine), # 9x9
#     'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine)
# }

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


# class StdConv(nn.Module):
#     """ Standard conv
#     ReLU - Conv - BN
#     """
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine)
#         )

#     def forward(self, x):
#         return self.net(x)


# class FacConv(nn.Module):
#     """ Factorized conv
#     ReLU - Conv(Kx1) - Conv(1xK) - BN
#     """
#     def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
#             nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine)
#         )

#     def forward(self, x):
#         return self.net(x)

# class DilConv(nn.Module):
#     """ (Dilated) depthwise separable conv
#     ReLU - (Dilated) depthwise separable - Pointwise - BN

#     If dilation == 2, 3x3 conv => 5x5 receptive field
#                       5x5 conv => 9x9 receptive field
#     """
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
#                       bias=False),
#             nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine)
#         )

#     def forward(self, x):
#         return self.net(x)


# class SepConv(nn.Module):
#     """ Depthwise separable conv
#     DilConv(dilation=1) * 2
#     """
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#         super().__init__()
#         self.net = nn.Sequential(
#             DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
#             DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
#         )

#     def forward(self, x):
#         return self.net(x)


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
        x = self.relu(x)
        
        first = self.conv11(x)
        first = self.conv12(first)
        second = self.conv21(x[:, :, 1:, 1:])
        second = self.conv22(second)

        out = torch.cat([first, second], dim=1)
        out = self.bn(out)
        return out

# class FactorizedReduce(nn.Module):
#     """
#     Reduce feature map size by factorized pointwise(stride=2).
#     """
#     def __init__(self, C_in, C_out, affine=True):
#         super().__init__()
#         self.relu = nn.ReLU()
#         self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#         self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#         self.bn = nn.BatchNorm2d(C_out, affine=affine)

#     def forward(self, x):
#         x = self.relu(x)
#         out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
#         out = self.bn(out)
#         return out


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
