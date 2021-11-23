import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
dbs=["agedb", "lfw", "calfw", "cplfw", "cfp", "megaface", "megafacer", "IJB-B", "IJB-C"]

for db in dbs:
    if (db=="agedb"):
        accuracies=[97.28 ,96.4, 98.15,
                    96.98, 97.05, 95.62,
                    97.6, 94.4, 97.3,
                    96.07, 93.22, 97.05,
                    96.63, 95.62, 96.83,
                    96.1, 96.35, 96.78, 97.17 ]
        params =   [4.5,  3.4,   5.0,
                    3.95, 3.07, 1.04,
                    2.0,   3.2,  2.6,
                    0.99,   0.5,  3.95,
                    3.07, 1.04,  1.35,
                    0.925, 0.99, 1.68,  1.75]
        nets=["ShuffleFaceNet 2x", "MobileFaceNetV1", "VarGFaceNet",
              "ShuffleMixFaceNet-M", "ShuffleMixFaceNet-S", "ShuffleMixFaceNet-XS",
              "MobileFaceNet", "ProxylessFaceNAS", "ShuffleFaceNet 1.5x",
              "MobileFaceNets", "ShuffleFaceNet 0.5x", "MixFaceNet-M",
              "MixFaceNet-S", "MixFaceNet-XS", "Distill-DSE-LSE",
              "PocketNetS-128 (ours)", "PocketNetS-256 (ours)", "PocketNetM-128 (ours)", "PocketNetM-256 (ours)"]

        marker=["X",'2', '<',
                "h", "h", "h",
                '1', 'x', '+',
                'D', '3' ,'v',
                'v', 'v', '*' ,
                'o', 'o', 'o','o']
        save_path = "./agedb.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(accuracies, params, 'o')
        plt.ylabel("Accuracy (%) ",fontsize=26)
        plt.xlabel("Params",fontsize=26)
        plt.ylim([90, 98])
        plt.xlim([0.8, 6])
    elif(db=="lfw"):
        accuracies = [99.62, 99.4, 99.85,
                      99.6, 99.58, 99.6,
                      99.7, 99.2, 99.7,
                      99.55, 99.23, 99.68,
                      99.6, 99.6, 99.67,
                      99.58, 99.66, 99.65, 99.58]
        params = [4.5, 3.4, 5.0,
                  3.95, 3.07, 1.04,
                  2.0, 3.2, 2.6,
                  0.99, 0.5, 3.95,
                  3.07, 1.04, 1.35,
                  0.925, 0.99, 1.68, 1.75]
        nets = ["ShuffleFaceNet 2x", "MobileFaceNetV1", "VarGFaceNet",
                "ShuffleMixFaceNet-M", "ShuffleMixFaceNet-S", "ShuffleMixFaceNet-XS",
                "MobileFaceNet", "ProxylessFaceNAS", "ShuffleFaceNet 1.5x",
                "MobileFaceNets", "ShuffleFaceNet 0.5x", "MixFaceNet-M",
                "MixFaceNet-S", "MixFaceNet-XS", "Distill-DSE-LSE",
                "PocketNetS-128 (ours)", "PocketNetS-256 (ours)", "PocketNetM-128 (ours)", "PocketNetM-256 (ours)"]

        marker = ["X", '2', '<',
                  "h", "h", "h",
                  '1', 'x', '+',
                  'D', '3', 'v',
                  'v', 'v', '*',
                  'o', 'o', 'o', 'o']
        save_path = "./lfw.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(accuracies, params, 'o')
        plt.ylabel("Accuracy (%) ",fontsize=26)
        plt.xlabel("Params",fontsize=26)
        plt.ylim([98, 99.89])
        plt.xlim([0.8, 6])
    elif(db=="calfw"):
        accuracies = [94.47, 95.15,
                      95.2, 92.55, 95.05,
                      95.63,
                      95.48, 95.5, 95.67, 95.63]
        params = [3.4, 5.0,
                  2.0, 3.2, 2.6,
                  1.35,
                  0.925, 0.99, 1.68, 1.75]
        nets = ["MobileFaceNetV1", "VarGFaceNet",
                "MobileFaceNet", "ProxylessFaceNAS", "ShuffleFaceNet 1.5x",
                "Distill-DSE-LSE",
                "PocketNetS-128 (ours)", "PocketNetS-256 (ours)", "PocketNetM-128 (ours)", "PocketNetM-256 (ours)"]

        marker = ['2', '<',
                  '1', 'x', '+',
                  '*',
                  'o', 'o', 'o', 'o']
        save_path = "./calfw.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(accuracies, params, 'o')
        plt.ylabel("Accuracy (%) ",fontsize=26)
        plt.xlabel("Params",fontsize=26)
        plt.ylim([86, 97])
        plt.xlim([0.8, 6])
    elif(db=="cplfw"):
        accuracies = [87.17, 88.55,
                      89.22, 84.17, 88.50,
                      89.68,
                      89.63, 88.93, 90, 90.03]
        params = [3.4, 5.0,
                  2.0, 3.2, 2.6,
                  1.35,
                  0.925, 0.99, 1.68, 1.75]
        nets = ["MobileFaceNetV1", "VarGFaceNet",
                "MobileFaceNet", "ProxylessFaceNAS", "ShuffleFaceNet 1.5x",
                "Distill-DSE-LSE",
                "PocketNetS-128 (ours)", "PocketNetS-256 (ours)", "PocketNetM-128 (ours)", "PocketNetM-256 (ours)"]

        marker = ['2', '<',
                  '1', 'x', '+',
                  '*',
                  'o', 'o', 'o', 'o']
        save_path = "./cplfw.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(accuracies, params, 'o')
        plt.ylabel("Accuracy (%) ",fontsize=26)
        plt.xlabel("Params",fontsize=26)
        plt.ylim([80, 91])
        plt.xlim([0.8, 6])


    elif(db=="cfp"):
        accuracies = [97.56, 95.8, 98.5,
                      96.9, 94.7, 96.9,
                      94.19, 92.59,
                      94.21, 93.34, 95.07, 95.56]
        params = [4.5, 3.4, 5.0,
                  2.0, 3.2, 2.6,
                  1.35, 0.5,
                  0.925, 0.99, 1.68, 1.75]
        nets = ["ShuffleFaceNet 2x", "MobileFaceNetV1", "VarGFaceNet",
                "MobileFaceNet", "ProxylessFaceNAS", "ShuffleFaceNet 1.5x",
                "Distill-DSE-LSE", "ShuffleFaceNet 0.5x",
                "PocketNetS-128 (ours)", "PocketNetS-256 (ours)", "PocketNetM-128 (ours)", "PocketNetM-256 (ours)"]

        marker = ["X", '2', '<',
                  '1', 'x', '+',
                  '*', '3',
                  'o', 'o', 'o', 'o']
        save_path = "./cfp.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(accuracies, params, 'o')
        plt.ylabel("Accuracy (%) ",fontsize=26)
        plt.xlabel("Params",fontsize=26)
        plt.ylim([80, 98])
        plt.xlim([0.8, 6])

    elif(db=="megaface"):
        accuracies = [91.3, 93.9,
                      94.24, 93.6, 89.24,
                      95.2, 82.8, 93,
                      90.16, 94.26,
                      92.23, 89.4,
                      90.54, 91.77, 92.45, 92.75]
        params = [3.4, 5.0,
                  3.95, 3.07, 1.04,
                  2.0, 3.2, 2.6,
                  0.99, 3.95,
                  3.07, 1.04,
                  0.925, 0.99, 1.68, 1.75]
        nets = ["MobileFaceNetV1", "VarGFaceNet",
                "ShuffleMixFaceNet-M", "ShuffleMixFaceNet-S", "ShuffleMixFaceNet-XS",
                "MobileFaceNet", "ProxylessFaceNAS", "ShuffleFaceNet 1.5x",
                "MobileFaceNets", "MixFaceNet-M",
                "MixFaceNet-S", "MixFaceNet-XS",
                "PocketNetS-128 (ours)", "PocketNetS-256 (ours)", "PocketNetM-128 (ours)", "PocketNetM-256 (ours)"]

        marker = ['2', '<',
                  "h", "h", "h",
                  '1', 'x', '+',
                  'D', 'v',
                  'v', 'v',
                  'o', 'o', 'o', 'o']
        save_path = "./megaface.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(accuracies, params, 'o')
        plt.ylabel("TAR at FAR1e–6 ",fontsize=26)
        plt.xlabel("Params",fontsize=26)
        plt.ylim([80, 96])
        plt.xlim([0.8, 6])
    elif(db=="megafacer"):
        accuracies = [93, 95.6,
                      95.22, 95.19, 91.03,
                      96.8, 84.8, 94.6,
                      92.59, 95.83,
                      93.79, 91.04,
                      92.23, 93.5, 94.17, 94.40]
        params = [3.4, 5.0,
                  3.95, 3.07, 1.04,
                  2.0, 3.2, 2.6,
                  0.99, 3.95,
                  3.07, 1.04,
                  0.925, 0.99, 1.68, 1.75]
        nets = ["MobileFaceNetV1", "VarGFaceNet",
                "ShuffleMixFaceNet-M", "ShuffleMixFaceNet-S", "ShuffleMixFaceNet-XS",
                "MobileFaceNet", "ProxylessFaceNAS", "ShuffleFaceNet 1.5x",
                "MobileFaceNets", "MixFaceNet-M",
                "MixFaceNet-S", "MixFaceNet-XS",
                "PocketNetS-128 (ours)", "PocketNetS-256 (ours)", "PocketNetM-128 (ours)", "PocketNetM-256 (ours)"]

        marker = ['2', '<',
                  "h", "h", "h",
                  '1', 'x', '+',
                  'D', 'v',
                  'v', 'v',
                  'o', 'o', 'o', 'o']
        save_path = "./megafacer.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(accuracies, params, 'o')
        plt.ylabel("TAR at FAR1e–6 ",fontsize=26)
        plt.xlabel("Params",fontsize=26)
        plt.ylim([80, 98])
        plt.xlim([0.8, 6])

    elif(db=="IJB-B"):
        accuracies = [92, 92.9,
                      91.47, 90.94, 87.86,
                      92.8, 87.1, 92.3,
                      91.55,
                      90.17, 88.48,
                      89.44, 89.31, 90.63, 90.74]
        params = [3.4, 5.0,
                  3.95, 3.07, 1.04,
                  2.0, 3.2, 2.6,
                  3.95,
                  3.07, 1.04,
                  0.925, 0.99, 1.68, 1.75]
        nets = ["MobileFaceNetV1", "VarGFaceNet",
                "ShuffleMixFaceNet-M", "ShuffleMixFaceNet-S", "ShuffleMixFaceNet-XS",
                "MobileFaceNet", "ProxylessFaceNAS", "ShuffleFaceNet 1.5x",
                "MixFaceNet-M",
                "MixFaceNet-S", "MixFaceNet-XS",
                "PocketNetS-128 (ours)", "PocketNetS-256 (ours)", "PocketNetM-128 (ours)", "PocketNetM-256 (ours)"]

        marker = ['2', '<',
                  "h", "h", "h",
                  '1', 'x', '+',
                  'v',
                  'v', 'v',
                  'o', 'o', 'o', 'o']
        save_path = "./ijbb.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(accuracies, params, 'o')
        plt.ylabel("TAR at FAR1e–4 ",fontsize=26)
        plt.xlabel("Params",fontsize=26)
        plt.ylim([69, 95])
        plt.xlim([0.8, 6])
    elif(db=="IJB-C"):
        accuracies = [93.9, 94.7,
                      91.47, 93.08, 90.43,
                      94.7, 89.7, 94.3,
                      93.42,
                      92.3, 90.73,
                      91.62, 91.33, 92.63, 92.7]
        params = [3.4, 5.0,
                  3.95, 3.07, 1.04,
                  2.0, 3.2, 2.6,
                  3.95,
                  3.07, 1.04,
                  0.925, 0.99, 1.68, 1.75]
        nets = ["MobileFaceNetV1", "VarGFaceNet",
                "ShuffleMixFaceNet-M", "ShuffleMixFaceNet-S", "ShuffleMixFaceNet-XS",
                "MobileFaceNet", "ProxylessFaceNAS", "ShuffleFaceNet 1.5x",
                "MixFaceNet-M",
                "MixFaceNet-S", "MixFaceNet-XS",
                "PocketNetS-128 (ours)", "PocketNetS-256 (ours)", "PocketNetM-128 (ours)", "PocketNetM-256 (ours)"]

        marker = ['2', '<',
                  "h", "h", "h",
                  '1', 'x', '+',
                  'v',
                  'v', 'v',
                  'o', 'o', 'o', 'o']
        save_path = "./ijbc.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(accuracies, params, 'o')
        plt.ylabel("TAR at FAR1e–4 ",fontsize=26)
        plt.xlabel("Params",fontsize=26)
        plt.ylim([69, 97])
        plt.xlim([0.8, 6])

    p=[]
    for i in range(len(accuracies)):
        if "ours" in nets[i]:
            plt.plot(params[i], accuracies[i], marker[i],markersize=16,markeredgecolor='red',label=nets[i])
        else:
            plt.plot(params[i], accuracies[i], marker[i],markersize=16,label=nets[i])

    plt.grid()
    plt.tight_layout()

    plt.legend(numpoints=1, loc='lower right',fontsize=12,ncol=2)
    plt.savefig(save_path, format='pdf', dpi=600)
    plt.close()
