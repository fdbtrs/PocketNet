""" CNN for network augmentation """
import torch
import torch.nn as nn
from models.augment_cells import AugmentCell
from models import ops

class AugmentCNN(nn.Module):
    """ Augmented CNN model """
    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype,
                 stem_multiplier=4,emb=128):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        # aux head position
        #self.aux_pos = 2*n_layers//3 if auxiliary else -1

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
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out

        # As in MobileFaceNet 
        # conv 1x1
        # linearGDConv 7x7
        # linear conv1x1
        con = 1024 if emb == 512 else 512

        self.tail = nn.Sequential(
            nn.PReLU(C_p),
            nn.Conv2d(C_p, con, 1, 1, 0, bias=False),
            nn.BatchNorm2d(con),
            nn.PReLU(con),
            nn.Conv2d(con, con, 7, 1, 0, groups=512, bias=False),
            nn.BatchNorm2d(con),
            nn.Conv2d(con, emb, 1, 1, 0, bias=False),
            nn.BatchNorm2d(emb)
        )

        self.dropout = nn.Dropout(0.1)

        self.linear = nn.Linear(emb, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)

        #aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            #if i == self.aux_pos and self.training:
            #    aux_logits = self.aux_head(s1)

        out = self.tail(s1)
        out = out.view(out.size(0), -1) # flatten

        out = self.dropout(out)

        logits = self.linear(out)

        norm = torch.norm(out, 2, 1, True)
        emb = torch.div(out, norm)

        return logits, emb # return classification output and embedding

    def l2_norm(input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p
