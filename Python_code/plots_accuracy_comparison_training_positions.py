
"""
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Palatino'
mpl.rcParams['text.usetex'] = 'true'
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Accent.colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receiver', help='Receiver id')
    parser.add_argument('model_name', help='Model name')
    parser.add_argument('model_type', help='convolutional, attention')
    args = parser.parse_args()

    ################################################
    # PLOTS ACCURACY COMPARISON TRAINING POSITIONS #
    ################################################
    # S3
    pos_train_val_s3 = [1, 2, 3, 4, 5]
    pos_test_s3 = [6, 7, 8, 9]

    # S2
    pos_train_val_s2 = [1, 3, 5, 7, 9]
    pos_test_s2 = [2, 4, 6, 8]

    # S1
    pos_train_val_s1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    pos_test_s1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    pos_train_val_complete = [pos_train_val_s1, pos_train_val_s2, pos_train_val_s3]
    pos_test_complete = [pos_test_s1, pos_test_s2, pos_test_s3]

    accuracy_s1 = []
    accuracy_s2 = []
    accuracy_s3 = []
    accuracy_sets = [accuracy_s1, accuracy_s2, accuracy_s3]
    for idx_set in range(len(pos_train_val_complete)):
        pos_train_set = pos_train_val_complete[idx_set]
        pos_test = pos_test_complete[idx_set]
        for num_pos_train in range(len(pos_train_set)):
            pos_train = pos_train_set[:num_pos_train+1]
            name_0 = args.model_name + 'IDrecs[\'57\']_TX[0, 1, 2]_RX[0]_posTRAIN' + str(pos_train) + '_posTEST' \
                     + str(pos_test) + '_bandwidth80_MOD' + args.model_type
            name_file = './outputs/' + name_0 + '.txt'
            with open(name_file, "rb") as fp:
                metrics_dict = pickle.load(fp)
            accuracy_sets[idx_set].append(metrics_dict['accuracy_test'])

    labels = [r'\texttt{S1}', r'\texttt{S2}', r'\texttt{S3}']

    accuracy_s1 = accuracy_sets[0]
    accuracy_s2 = accuracy_sets[1]
    accuracy_s3 = accuracy_sets[2]

    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(9, 4)
    x = np.arange(9)
    width = 0.25
    rects1 = plt.bar(x[:4], accuracy_s1[:4], width, edgecolor='k', linestyle='--')
    plt.gca().set_prop_cycle(None)
    rects1 = plt.bar(x[4:] - width, accuracy_s1[4:], width,
                     label=labels[0], edgecolor='k', linestyle='--')
    rects2 = plt.bar(x[4:], accuracy_s2, width,
                     label=labels[1],
                     edgecolor='k', linestyle='--')  # $N_{\rm row}=$~
    rects3 = plt.bar(x[4:] + width, accuracy_s3, width,
                     label=labels[2],
                     edgecolor='k', linestyle='--')
    plt.grid()
    plt.legend(fontsize='medium', ncol=1)
    plt.xlabel(r'number of training positions', fontsize=18)
    plt.ylabel(r'Accuracy [\%]', fontsize=18)
    plt.xticks(x, np.flip(x) + 1)
    name_fig = './plots/accuracy_comparison_train_pos.pdf'
    plt.savefig(name_fig)
    plt.close()
