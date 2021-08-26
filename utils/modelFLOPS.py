from backbones.augment_cnn import AugmentCNN
from backbones import genotypes as gt

from pytorch_model_summary import summary
from util.config import config as cfg

import torch
from torch.autograd import Variable
import numpy as np

from backbones.iresnet import iresnet100




def count_model_flops(model, input_res=[112, 112], multiply_adds=True):
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (
            2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        if self.bias is not None:
            bias_ops = self.bias.nelement() if self.bias.nelement() else 0
            flops = batch_size * (weight_ops + bias_ops)
        else:
            flops = batch_size * weight_ops
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)
    def pooling_hook_ad(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        input = input[0]
        flops =  int(np.prod(input.shape))
        list_pooling.append(flops)

    handles = []

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.ConvTranspose2d):
                handles.append(net.register_forward_hook(conv_hook))
            elif isinstance(net, torch.nn.Linear):
                handles.append(net.register_forward_hook(linear_hook))
            elif isinstance(net, torch.nn.BatchNorm2d) or isinstance(net, torch.nn.BatchNorm1d):
                handles.append(net.register_forward_hook(bn_hook))
            elif isinstance(net, torch.nn.ReLU) or isinstance(net, torch.nn.PReLU) or isinstance(net,torch.nn.Sigmoid) or isinstance(net, HSwish) or isinstance(net, Swish):
                handles.append(net.register_forward_hook(relu_hook))
            elif isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                handles.append(net.register_forward_hook(pooling_hook))
            elif isinstance(net, torch.nn.AdaptiveAvgPool2d):
                handles.append(net.register_forward_hook(pooling_hook_ad))
            else:
                print("warning" + str(net))
            return
        for c in childrens:
            foo(c)

    model.eval()
    foo(model)
    input = Variable(torch.rand(3, input_res[1], input_res[0]).unsqueeze(0), requires_grad=True)
    out = model(input)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    for h in handles:
        h.remove()
    model.train()
    return flops_to_string(total_flops)

def flops_to_string(flops, units='MFLOPS', precision=4):
    if units == 'GFLOPS':
        return str(round(flops / 10.**9, precision)) + ' ' + units
    elif units == 'MFLOPS':
        return str(round(flops / 10.**6, precision)) + ' ' + units
    elif units == 'KFLOPS':
        return str(round(flops / 10.**3, precision)) + ' ' + units
    else:
        return str(flops) + ' FLOPS'


if __name__ == "__main__":
    genotype = gt.from_str(cfg.genotypes["softmax_cifar10"])

    model = AugmentCNN(112, 3, 16, 18, genotype, stem_multiplier=4, emb=128)
    model = iresnet100(num_features=128)
    print(model)

    print(summary(model, torch.zeros((1, 3, 112, 112)), show_input=False))

    flops = count_model_flops(model)

    print(flops)

    # model.eval()
    # tic = time.time()

    # model.forward(torch.zeros((1, 3, 112, 112)))
    # end = time.time()
    # print(end-tic)