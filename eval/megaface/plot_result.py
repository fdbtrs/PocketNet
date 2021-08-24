# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 08:07:14 2018

@author: zhaoy
"""
import os
import os.path as osp
import numpy as np
import json
from fnmatch import fnmatch
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from megaface.megaface_interpolation import linear_interp, linear_interp_logx

sample='DartFaceNet256-sx2-KD2-26-noisy' # -------> add megaface path here


def generate_n_distractors():
    n_distractors = [10 ** i for i in range(1, 7)]
    n_distractors = [1000000]

    return n_distractors


n_distractors = generate_n_distractors()


def generate_plot_colors():
    _colors = ['w', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

    rgb_colors_list = []

    # 9 basic colors
    for it in _colors:
        rgb_colors_list.append(matplotlib.colors.to_rgb(it))

    # add other colors not in the 9 basic colors
    for r in (0, 0.5, 1.0):
        for g in (0, 0.5, 1.0):
            for b in (0, 0.5, 1.0):
                rgb = (r, g, b)
                if rgb not in rgb_colors_list:
                    rgb_colors_list.append(rgb)

    # remove white color
    rgb_colors_list.remove((1.0, 1.0, 1.0))
    print('===> {} colors generated without "white" color'.format(
        len(rgb_colors_list)))

    return rgb_colors_list


colors = generate_plot_colors()


def interp_target_tpr(roc, target_fpr):
    if (target_fpr < roc[0][0] or target_fpr > roc[0][-1]):
        print('target_fpr out of bound, will return -1')
        return -1.0

    # # This interpolation might be the one that MegaFace officially uses
    # for i, fpr in enumerate(roc[0]):
    #     if fpr > target_fpr:
    #         return roc[1][i]

    # linear interpolation
    for i, fpr in enumerate(roc[0]):
        if fpr > target_fpr:
            break

    # linear x interpolation
    # target_tpr = linear_interp(target_fpr,
    #                              roc[0][i - 1], roc[0][i],
    #                              roc[1][i - 1], roc[1][i]
    #                              )

    # linear logx interpolation
    target_tpr = linear_interp_logx(target_fpr,
                                    roc[0][i - 1], roc[0][i],
                                    roc[1][i - 1], roc[1][i]
                                    )

    # NN interpolation
    # target_tpr = nearest_neighbor_interp(target_fpr, roc[0], roc[1])

    return target_tpr


def interp_target_rank_recall(cmc, target_rank):
    if (target_rank < cmc[0][0] or target_rank > cmc[0][-1]):
        print('target_fpr out of bound, will return -1')
        return -1.0

    # This interpolation might be the one that MegaFace officially uses
    # for i, fpr in enumerate(cmc[0]):
    #     if fpr > target_rank:
    #         return cmc[1][i]

    # linear interpolation
    for i, rank in enumerate(cmc[0]):
        if rank > target_rank:
            break
    if cmc[0][i - 1] == target_rank:
        target_recall = cmc[1][i - 1]
    else:
        target_recall = linear_interp(target_rank,
                                      cmc[0][i - 1], cmc[0][i],
                                      cmc[1][i - 1], cmc[1][i]
                                      )

    # NN interpolation
    # target_recall = nearest_neighbor_interp(target_rank, cmc[0], cmc[1])

    return target_recall


def load_result_data(folder, probe_name):
    #    n_distractors = generate_n_distractors()
    print('===> Load result data from ', folder)

    all_files = os.listdir(folder)
    #    print 'all_files: ', all_files
    cmc_files = sorted(
        [a for a in all_files if fnmatch(a.lower(), 'cmc*%s*_1.json' % probe_name.lower())])[::-1]
    #    print 'cmc_files: ', cmc_files

    if not cmc_files:
        return None

    cmc_dict = {}
    for i, filename in enumerate(cmc_files):
        with open(os.path.join(folder, filename), 'r') as f:
            cmc_dict[n_distractors[i]] = json.load(f)

    rocs = []

    print('cmc_dict', cmc_dict)
    print('n_distractors', n_distractors)

    for i in n_distractors:
        rocs.append(cmc_dict[i]['roc'])

    cmcs = []

    for i in n_distractors:
        for j in range(len(cmc_dict[i]['cmc'][0])):
            cmc_dict[i]['cmc'][0][j] += 1

        cmcs.append(cmc_dict[i]['cmc'])

    rank_1 = [cmc_dict[n]['cmc'][1][0]
              for n in n_distractors]

    rank_10 = []
    for i in range(len(n_distractors)):
        target_recall = interp_target_rank_recall(cmcs[i], 10)
        rank_10.append(target_recall)

    # roc_10K = cmc_dict[10000]['roc']
    # roc_100K = cmc_dict[100000]['roc']
    # roc_1M = cmc_dict[1000000]['roc']

    return {
        'rocs': rocs,
        'cmcs': cmcs,
        'Rank_1': rank_1,
        'Rank_10': rank_10
        # 'roc_10k': roc_10K,
        # 'roc_100k': roc_100K,
        # 'roc_1M': roc_1M
    }


def calc_target_tpr_and_rank(rocs, rank_1, rank_10, save_dir,
                             method_label=None, target_fpr=1e-6,
                             fp_tpr_sum=None, fp_rank_sum=None):
    print('===> Calc and save TPR@FPR={:g} for method: {}'.format(
        target_fpr, method_label))
    fn_tpr = osp.join(save_dir, 'TPRs-at-FPR_%g' % target_fpr)
    fn_rank = osp.join(save_dir, 'Rank_vs_distractors')

    if method_label:
        fn_tpr += '_' + method_label
        fn_rank += '_' + method_label
    else:
        method_label = "YOUR Method"

    fn_tpr += '.txt'
    fn_rank += '.txt'

    fp_tpr = open(fn_tpr, 'w')

    write_string_sum = ''
    if fp_tpr_sum:
        write_string_sum += '{:32}'.format(method_label)

    write_string = 'TPR@FPR=%g at different #distractors\n' % target_fpr
    write_string += '#distractors  TPR\n'
    print(write_string)
    fp_tpr.write(write_string)

    for i, roc in enumerate(rocs):
        target_tpr = interp_target_tpr(roc, target_fpr)
        write_string = '%7d %7.6f\n' % (n_distractors[i], target_tpr)
        print(write_string)
        fp_tpr.write(write_string)

        if fp_tpr_sum:
            write_string_sum += "\t{:<9.6f}".format(target_tpr)

    if fp_tpr_sum:
        fp_tpr_sum.write(write_string_sum + '\n')
        fp_tpr_sum.flush()

    fp_tpr.close()

    print('===> Save Rank_1 under different #distractors for method: ', method_label)
    fp_rank = open(fn_rank, 'w')

    write_string_sum = ''
    if fp_rank_sum:
        write_string_sum += '{:32}'.format(method_label)

    write_string = 'Rank_1 recall at different #distractors\n'
    write_string += '#distractors  recall\n'
    print(write_string)
    fp_rank.write(write_string)

    for i, rank in enumerate(rank_1):
        write_string = '%7d  %7.6f\n' % (n_distractors[i], rank)
        print(write_string)
        fp_rank.write(write_string)
        if fp_rank_sum:
            write_string_sum += "\t{:<16.6f}".format(rank)

    write_string = '\nRank_10 recall at different #distractors\n'
    write_string += '#distractors  recall\n'
    print(write_string)
    fp_rank.write(write_string)

    for i, rank in enumerate(rank_10):
        write_string = '%7d  %7.6f\n' % (n_distractors[i], rank)
        print(write_string)
        fp_rank.write(write_string)
        if fp_rank_sum:
            write_string_sum += "\t{:<16.6f}".format(rank)

    if fp_rank_sum:
        fp_rank_sum.write(write_string_sum + '\n')
        fp_rank_sum.flush()

    fp_rank.close()


# %matplotlib inline
def plot_megaface_result(your_method_dirs, your_method_labels,
                         probe_name,
                         save_dir=None,
                         other_methods_dir=None,
                         save_tpr_and_rank1_for_others=False,
                         ymin=0, minor_ticks=5,
                         target_fpr=1e-6,
                         show_plot=False):
    probe_name = probe_name.lower()
    valid_probe_names = ['facescrub', 'fgnet', 'idprobe']
    if not probe_name in valid_probe_names:
        raise Exception(
            'probeset name must be one of {}!'.format(valid_probe_names))

    if not save_dir:
        save_dir = './rlt_%s_results' % (probe_name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    #    n_distractors = generate_n_distractors()

    print('n_distractors: ', n_distractors)

    if your_method_dirs is None:
        your_method_dirs = []

    if your_method_labels is None:
        your_method_labels = []
        for it in your_method_dirs:
            your_method_labels.append(osp.basename(it))

    your_methods_data = []

    fn_tpr_sum = osp.join(
        save_dir, 'TPRs-at-FPR_%g_summary_all.txt' % target_fpr)
    fn_rank_sum = osp.join(save_dir, 'Rank_vs_distractors_summary_all.txt')

    fp_tpr_sum = open(fn_tpr_sum, 'w')
    fp_rank_sum = open(fn_rank_sum, 'w')

    # write table head for TPR summary
    write_string = 'TPR@FPR={:g} at different #distractors\n\n'.format(
        target_fpr)
    fp_tpr_sum.write(write_string)
    write_string = '{:32}'.format('method')
    for it in n_distractors:
        write_string += '\t{:<7d}'.format(it)
    write_string += '\n'
    fp_tpr_sum.write(write_string)

    # write table head for Rank summary
    write_string = 'Rank-1 and Rank-10 at different #distractors\n\n'
    fp_rank_sum.write(write_string)
    write_string = '{:32}'.format('method')
    for it in n_distractors:
        write_string += '\trank1@{:<8d}'.format(it)
    for it in n_distractors:
        write_string += '\trank10@{:<7d}'.format(it)
    write_string += '\n'
    fp_rank_sum.write(write_string)

    n_results = len(your_method_dirs)
    for j in range(n_results):
        print('===> Loading data for probset {} from: {}'.format(
            probe_name, your_method_dirs[j]))

        # your_result = load_your_result(your_method_dirs, probe_name, feat_ending)
        your_result = load_result_data(your_method_dirs[j], probe_name)

        rocs = your_result['rocs']
        cmcs = your_result['cmcs']
        rank_1 = your_result['Rank_1']
        rank_10 = your_result['Rank_10']

        your_methods_data.append(your_result)

        calc_target_tpr_and_rank(rocs, rank_1, rank_10,
                                 save_dir, your_method_labels[j],
                                 target_fpr,
                                 fp_tpr_sum=fp_tpr_sum,
                                 fp_rank_sum=fp_rank_sum)

        print('===> Plotting Verification ROC under different #distractors')
        fig = plt.figure(figsize=(16, 12), dpi=100)

        labels = [str(it) for it in n_distractors]

        # plt.semilogx(rocs[0][0], rocs[0][1], 'g', label='10')
        # plt.semilogx(rocs[1][0], rocs[1][1], 'r', label='100')
        # plt.semilogx(rocs[2][0], rocs[2][1], 'b', label='1000')
        # plt.semilogx(your_result['roc_10k'][0],
        #              your_result['roc_10k'][1], 'c', label='10000')
        # plt.semilogx(rocs[4][0], rocs[4][1], 'm', label='100000')
        # plt.semilogx(your_result['roc_1M'][0],
        #              your_result['roc_1M'][1], 'y', label='1000000')

        color_idx = 0
        for i in range(len(n_distractors)):
            if color_idx < len(colors):
                _color = colors[color_idx]
            else:
                _color = np.random.rand(3)

            plt.semilogx(rocs[i][0], rocs[i][1], c=_color, label=labels[i])
            color_idx += 1

        plt.xlim([1e-8, 1])
        plt.ylim([ymin, 1])

        plt.grid(True, which='major', lw=2)

        if minor_ticks > 0:
            ax = plt.gca()
            # minorLocator = AutoMinorLocator(minor_ticks)
            # ax.xaxis.set_minor_locator(minorLocator)
            minorLocator = AutoMinorLocator(minor_ticks)
            ax.yaxis.set_minor_locator(minorLocator)
            plt.grid(True, which='minor', ls='--')

        # plt.grid(True, which='both')

        ax.set_xlabel('FPR (log scale)')
        ax.set_ylabel('TPR')

        plt.legend(loc='lower right')
        if show_plot:
            plt.show()
        save_fn = osp.join(save_dir,
                           'roc_under_diff_distractors_%s.png' % your_method_labels[j])
        fig.savefig(save_fn, bbox_inches='tight')

        print('===> Plotting Identification CMC under different #distractors')
        fig = plt.figure(figsize=(16, 12), dpi=100)

        color_idx = 0
        for i in range(len(n_distractors)):
            if color_idx < len(colors):
                _color = colors[color_idx]
            else:
                _color = np.random.rand(3)

            plt.semilogx(cmcs[i][0], cmcs[i][1], c=_color, label=labels[i])
            color_idx += 1

        plt.xlim([1, 1e6])
        plt.ylim([ymin, 1])

        plt.grid(True, which='major', lw=2)

        if minor_ticks > 0:
            ax = plt.gca()
            # minorLocator = AutoMinorLocator(minor_ticks)
            # ax.xaxis.set_minor_locator(minorLocator)
            minorLocator = AutoMinorLocator(minor_ticks)
            ax.yaxis.set_minor_locator(minorLocator)
            plt.grid(True, which='minor', ls='--')

        # plt.grid(True, which='both')

        ax.set_xlabel('Rank (log scale)')
        ax.set_ylabel('Identification Rate')

        plt.legend(loc='lower right')
        if show_plot:
            plt.show()
        save_fn = osp.join(save_dir, 'cmc_under_diff_distractors_%s.png'
                           % your_method_labels[j])
        fig.savefig(save_fn, bbox_inches='tight')

    print('===> Load result data for all the other methods')
    other_methods_list = []
    other_methods_data = {}

    if other_methods_dir:
        other_methods_list = os.listdir(other_methods_dir)
        print('===> other_methods_list before cleaning: ', other_methods_list)

        for it in other_methods_list:
            if not osp.isdir(osp.join(other_methods_dir, it)):
                print("Remove flies(not folders) from other_methods_list")
                other_methods_list.remove(it)

        for dd in your_method_dirs:
            dd_dir, dd_base = osp.split(osp.realpath(dd))
            #            print '---> dd_dir, dd_base: ', dd_dir, dd_base
            #            print osp.realpath(other_methods_dir)
            if dd_dir and dd_dir == osp.realpath(other_methods_dir):
                for it in other_methods_list:
                    #                    print it
                    if dd_base == it:
                        print("Remove your_method_dirs from other_methods_list")
                        other_methods_list.remove(it)

    print('===> other_methods_list after cleaning: ', other_methods_list)

    if not (your_method_dirs or other_methods_list):
        print('===> No valid methods found, neither yours or others.')
        pass

    # ['3divi',
    #  'deepsense',
    #  'ntech',
    #  'faceall_norm',
    #  'Vocord',
    #  'Barebones_FR',
    #  'ntech_small',
    #  'deepsense_small',
    #  'SIAT_MMLAB',
    #  'faceall',
    #  'facenet',
    #  'ShanghaiTech']

    if other_methods_list:
        other_methods_data = {}

        for method in other_methods_list:
            result_data = load_result_data(
                os.path.join(other_methods_dir, method), probe_name)

            if result_data is not None:
                other_methods_data[method] = load_result_data(
                    os.path.join(other_methods_dir, method), probe_name)
        other_methods_list = other_methods_data.keys()

        if save_tpr_and_rank1_for_others:
            for name in other_methods_list:
                calc_target_tpr_and_rank(other_methods_data[name]['rocs'],
                                         other_methods_data[name]['Rank_1'],
                                         other_methods_data[name]['Rank_10'],
                                         save_dir, name,
                                         target_fpr,
                                         fp_tpr_sum=fp_tpr_sum,
                                         fp_rank_sum=fp_rank_sum)

    plot_names = ['10k', '100k', '1M']
    # plot_names = ['1M']

    for i, pn in enumerate(plot_names):
        print('===> Plotting ROC under %s distractors for your methods' % pn)
        fig = plt.figure(figsize=(20, 10), dpi=200)
        ax = plt.subplot(111)

        color_idx = 0

        for j in range(n_results):
            if color_idx < len(colors):
                _color = colors[color_idx]
            else:
                _color = np.random.rand(3)

            #ax.semilogx(your_methods_data[j]['rocs'][i + 3][0],
            #            your_methods_data[j]['rocs'][i + 3][1],
            #            label=your_method_labels[j],
            #            c=_color)
            color_idx += 1

        if other_methods_list:
            print('===> Plotting ROC under %s distractors for all the other methods' % pn)

            for name in other_methods_list:
                if color_idx < len(colors):
                    _color = colors[color_idx]
                else:
                    _color = np.random.rand(3)

                ax.semilogx(other_methods_data[name]['rocs'][i + 3][0],
                            other_methods_data[name]['rocs'][i + 3][1],
                            label=name,
                            c=_color)
                color_idx += 1

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.set_xlim([1e-6, 1])
        ax.set_ylim([ymin, 1])

        ax.set_xlabel('FPR (log scale)')
        ax.set_ylabel('TPR')

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.rcParams['figure.figsize'] = (
            10.0, 8.0)  # set default size of plots

        plt.grid(True, which='major', lw=2)

        if minor_ticks > 0:
            # ax = plt.gca()
            # minorLocator = AutoMinorLocator(minor_ticks)
            # ax.xaxis.set_minor_locator(minorLocator)
            minorLocator = AutoMinorLocator(minor_ticks)
            ax.yaxis.set_minor_locator(minorLocator)
            plt.grid(True, which='minor', ls='--')

        # plt.grid(True, which='both')
        #    plt.legend()
        if show_plot:
            plt.show()
        fig.savefig(osp.join(save_dir, 'verification_roc_%s.png' % pn),
                    bbox_inches='tight')

        print('===> Plotting recall vs rank under %s distractors for your methods' % pn)
        fig = plt.figure(figsize=(20, 10), dpi=200)
        ax = plt.subplot(111)
        color_idx = 0

        for j in range(n_results):
            if color_idx < len(colors):
                _color = colors[color_idx]
            else:
                _color = np.random.rand(3)

            ax.semilogx(your_methods_data[j]['cmcs'][i + 3][0],
                        your_methods_data[j]['cmcs'][i + 3][1],
                        label=your_method_labels[j],
                        c=_color)
            color_idx += 1

        if other_methods_list:
            print('===> Plotting recall vs rank under %s distractors for all the other methods' % pn)

            for name in other_methods_list:
                ax.semilogx(other_methods_data[name]['cmcs'][i + 3][0],
                            other_methods_data[name]['cmcs'][i + 3][1],
                            label=name,
                            c=np.random.rand(3))

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.set_xlim([1, 1e4])
        ax.set_ylim([ymin, 1])

        ax.set_xlabel('Rank (log scale)')
        ax.set_ylabel('Identification Rate')

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.rcParams['figure.figsize'] = (
            10.0, 8.0)  # set default size of plots

        plt.grid(True, which='major', lw=2)

        if minor_ticks > 0:
            # ax = plt.gca()
            # minorLocator = AutoMinorLocator(minor_ticks)
            # ax.xaxis.set_minor_locator(minorLocator)
            minorLocator = AutoMinorLocator(minor_ticks)
            ax.yaxis.set_minor_locator(minorLocator)
            plt.grid(True, which='minor', ls='--')

        # plt.grid(True, which='both')

        #    plt.legend()
        if show_plot:
            plt.show()
        fig.savefig(osp.join(save_dir, 'identification_recall_vs_rank_%s.png' % pn),
                    bbox_inches='tight')

    print('===> Plotting rank_1 vs #distractors for your method')
    fig = plt.figure(figsize=(20, 10), dpi=100)
    ax = plt.subplot(111)

    color_idx = 0
    for j in range(n_results):
        if color_idx < len(colors):
            _color = colors[color_idx]
        else:
            _color = np.random.rand(3)

        ax.semilogx(n_distractors, your_methods_data[j]['Rank_1'],
                    label=your_method_labels[j],
                    c=_color)
        color_idx += 1

    if other_methods_list:
        print('===> Plotting rank_1 vs #distractors for all the other methods')

        for name in other_methods_list:
            if color_idx < len(colors):
                _color = colors[color_idx]
            else:
                _color = np.random.rand(3)
            ax.semilogx(
                n_distractors,
                other_methods_data[name]['Rank_1'],
                label=name,
                c=_color)
            color_idx += 1

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.set_xlabel('#distractors (log scale)')
    ax.set_ylabel('Identification Rate (Rank-1)')

    ax.set_xlim([10, 1e6])
    ax.set_ylim([ymin, 1])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots

    plt.grid(True, which='major', lw=2)

    if minor_ticks > 0:
        # ax = plt.gca()
        # minorLocator = AutoMinorLocator(minor_ticks)
        # ax.xaxis.set_minor_locator(minorLocator)
        minorLocator = AutoMinorLocator(minor_ticks)
        ax.yaxis.set_minor_locator(minorLocator)
        plt.grid(True, which='minor', ls='--')

    # plt.grid(True, which='both')

    #    plt.legend()
    if show_plot:
        plt.show()
    fig.savefig(osp.join(save_dir, 'identification_rank_1_vs_distractors.png'),
                bbox_inches='tight')

    fp_tpr_sum.close()
    fp_rank_sum.close()


if __name__ == '__main__':
    
    your_method_dirs = [
        f'{sample}/', # Megaface
    ]
    your_method_labels = [
        'cos_s{}'.format(str(sample))
    ]

    probesets = ['facescrub']
    # feat_ending = '_feat'

    other_methods_dir = None
    save_tpr_and_rank1_for_others = False

    y_min = 0
    minor_ticks = 5
    target_fpr = 1e-6
    show_plot = False

    for probe_name in probesets:
        save_dir = './cmc_roc_results/{}'.format(str(sample))
        plot_megaface_result(your_method_dirs, your_method_labels,
                             probe_name,
                             save_dir,
                             other_methods_dir,
                             save_tpr_and_rank1_for_others,
                             y_min, minor_ticks,
                             show_plot=show_plot
                             )
