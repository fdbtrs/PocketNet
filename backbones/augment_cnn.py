""" CNN for network augmentation """
import torch
import torch.nn as nn
from backbones.augment_cells import AugmentCell
from backbones import ops
import math
import torch.nn.functional as F
from torch.autograd import Variable

class AugmentCNN(nn.Module):
    """ Augmented CNN model """
    def __init__(self, input_size=112, C_in=3, C=16, n_layers=18, genotype=None,
                 stem_multiplier=4, emb=128):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_layers = n_layers
        self.genotype = genotype

        C_cur = stem_multiplier * C

        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 2, 1, bias=False),
            nn.BatchNorm2d(C_cur),
        )

        # 
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        # generate each layer(cell) and store them in ModuleList
        for i in range(n_layers):
            # place reduction cell in 1/3 and 2/3 and 3/3 of network
            # else normal cell
            #if i in [n_layers//3, 2*n_layers//3]:
            if i in [n_layers//3, 2*n_layers//3, n_layers-1]:
            #if i in [n_layers//4, 2*n_layers//4, 3*n_layers//4]: # maybe interesting to put reduction cells elsewhere
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out

        con = 1024 if emb == 512 else 512
        # As in MobileFaceNet 
        # conv 1x1
        # linearGDConv 7x7
        # linear conv1x1
        self.tail = nn.Sequential(
            nn.PReLU(C_p),
            nn.Conv2d(C_p, con, 1, 1, 0, bias=False),
            nn.BatchNorm2d(con),
            nn.PReLU(con),
            nn.Conv2d(con, con, 7, 1, 0, groups=con, bias=False),
            nn.BatchNorm2d(con),
            nn.Conv2d(con, emb, 1, 1, 0, bias=False), # different embedding sizes 112/256/512
            nn.BatchNorm2d(emb)
        )
        
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d): # pool
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


    def forward(self, x):
        s0 = s1 = self.stem(x)

        #aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)

        out = self.tail(s1)
        out = out.view(out.size(0), -1) # flatten

        return out
if __name__ == "__main__":
    print("main")

