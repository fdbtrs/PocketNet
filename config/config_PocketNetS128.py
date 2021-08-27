from easydict import EasyDict as edict

config = edict()
config.dataset = "emoreKD"
config.embedding_size = 128
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1
config.output = "output/PocketNetS-128"
config.scale=1.0
config.global_step=0
config.s=64.0
config.m=0.5

config.genotypes = dict({
        "softmax_cifar10": "Genotype(normal=[[('dw_conv_7x7', 0), ('dw_conv_3x3', 1)], [('dw_conv_1x1', 1), ('dw_conv_1x1', 2)], [('max_pool_3x3', 2), ('dw_conv_7x7', 3)], [('dw_conv_5x5', 4), ('max_pool_3x3', 0)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('dw_conv_7x7', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))",
        "softmax_casia": "Genotype(normal=[[('dw_conv_3x3', 0), ('dw_conv_1x1', 1)], [('dw_conv_3x3', 2), ('dw_conv_5x5', 0)], [('dw_conv_3x3', 3), ('dw_conv_3x3', 0)], [('dw_conv_3x3', 4), ('skip_connect', 0)]], normal_concat=range(2, 6), reduce=[[('dw_conv_3x3', 1), ('dw_conv_7x7', 0)], [('skip_connect', 2), ('dw_conv_5x5', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))"    })

# for KD
config.teacher_pth = "output/iresnet128"
config.teacher_global_step = 295672
# if use pretrained model (not for resume!)
config.student_pth = ""
config.student_global_step = 0
config.net_name="PocketNetS"
if (config.net_name=="PocketNetS"):
    config.channel=16
    config.n_layers=18
elif (config.net_name=="PocketNetM"):
    config.channel = 32
    config.n_layers = 9

config.w=100

if config.dataset == "emoreKD":
    config.rec = "data/faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch = 26
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step = 5686

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14, 20, 25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = "data/faces_webface_112x112"
    config.num_classes = 10572
    config.num_image = 501195
    config.num_epoch = 34
    config.warmup_epoch = -1
    config.val_targets =  ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    config.lr_func = lr_step_func
