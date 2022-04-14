
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
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Accent.colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receiver', help='Receiver id')
    parser.add_argument('model_name', help='Model name')
    parser.add_argument('model_type', help='convolutional, attention')
    args = parser.parse_args()

    #######################################
    # PLOTS ACCURACY COMPARISON BANDWIDTH #
    #######################################
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

    accuracy_20 = []
    accuracy_40 = []
    accuracy_80 = []
    for idx_set in range(len(pos_train_val_complete)):
        pos_train = pos_train_val_complete[idx_set]
        pos_test = pos_test_complete[idx_set]
        name_20 = args.model_name + 'IDrecs[\'57\']_TX[0, 1, 2]_RX[0]_posTRAIN' + str(pos_train) + '_posTEST' \
                  + str(pos_test) + '_bandwidth20_MOD' + args.model_type
        name_file = './outputs/' + name_20 + '.txt'
        with open(name_file, "rb") as fp:
            metrics_dict = pickle.load(fp)
        accuracy_20.append(metrics_dict['accuracy_test'])

        name_40 = args.model_name + 'IDrecs[\'57\']_TX[0, 1, 2]_RX[0]_' \
                  'posTRAIN' + str(pos_train) + '_posTEST' \
                  + str(pos_test) + '_bandwidth40_MOD' + args.model_type
        name_file = './outputs/' + name_40 + '.txt'
        with open(name_file, "rb") as fp:
            metrics_dict = pickle.load(fp)
        accuracy_40.append(metrics_dict['accuracy_test'])

        name_80 = args.model_name + 'IDrecs[\'57\']_TX[0, 1, 2]_RX[0]_' \
                  'posTRAIN' + str(pos_train) + '_posTEST' \
                  + str(pos_test) + '_bandwidth80_MOD' + args.model_type
        name_file = './outputs/' + name_80 + '.txt'
        with open(name_file, "rb") as fp:
            metrics_dict = pickle.load(fp)
        accuracy_80.append(metrics_dict['accuracy_test'])

    labels = [r'\texttt{S1}', r'\texttt{S2}', r'\texttt{S3}']
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(6, 5)
    x = np.arange(len(labels))
    width = 0.25
    rects1 = plt.bar(x - width, accuracy_80, width,
                     label=r'80~MHz', edgecolor='k', linestyle='--')
    rects2 = plt.bar(x, accuracy_40, width,
                     label=r'40~MHz',
                     edgecolor='k', linestyle='--')
    rects3 = plt.bar(x + width, accuracy_20, width,
                     label=r'20~MHz',
                     edgecolor='k', linestyle='--')
    plt.grid(axis='y')
    plt.legend(fontsize='medium', ncol=1)
    plt.xlabel(r'set', fontsize=26)
    plt.ylabel(r'Accuracy [\%]', fontsize=26)
    plt.xticks(x, labels)
    name_fig = './plots/accuracy_comparison_bandwidth.pdf'
    plt.savefig(name_fig)
    plt.close()

    a = 1


