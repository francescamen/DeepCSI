
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
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Palatino'
mpl.rcParams['text.usetex'] = 'true'
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
mpl.rcParams['font.size'] = 26
mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Accent.colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receiver', help='Receiver id')
    parser.add_argument('model_name', help='Model name')
    parser.add_argument('model_type', help='convolutional, attention')
    args = parser.parse_args()

    ######################################
    # PLOTS VARYING THE NUMBER OF LAYERS #
    ######################################
    models_params = ['128,128,128,128,128,128,128-7,7,7,7,7,5,3',
                     '128,128,128,128,128,128-7,7,7,7,5,3',
                     '128,128,128,128,128-7,7,7,5,3',
                     '128,128,128,128-7,7,7,5',
                     '128,128,128-7,7,7',
                     '128,128-7,7']

    trainable_parameters_list = []
    accuracy_val_list = []

    for mod_idx in range(len(models_params)):
        model_n_params = models_params[mod_idx]
        model_n = './outputs/' + args.model_name + model_n_params + '_IDrecs[\'57\']_TX[0, 1, 2]_RX[0]_' \
                  'posTRAIN[1, 2, 3, 4, 5, 6, 7, 8, 9]_' \
                  'posTEST[1, 2, 3, 4, 5, 6, 7, 8, 9]_bandwidth80_' \
                  'MOD' + args.model_type + '_hyper_selection-' + model_n_params + '.txt'

        with open(model_n, "rb") as fp:
            metrics_dict = pickle.load(fp)

        trainable_parameters_list.append(metrics_dict['trainable_parameters'])
        accuracy_val_list.append(metrics_dict['accuracy_val'])

    num_layers = [7, 6, 5, 4, 3, 2]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(6, 5)
    plt.plot(trainable_parameters_list, accuracy_val_list, marker='o', markersize=16)
    plt.grid(axis='y')
    plt.grid(axis='x')
    plt.xlabel(r'trainable params', fontsize=30)
    plt.ylabel(r'Accuracy [\%]', fontsize=30)
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
    # plt.annotate('7,6,5', (trainable_parameters_list[0] - 5e4, accuracy_val_list[0] + 9e-3), fontsize=24)
    for iii, n_lay in enumerate(num_layers):
        plt.annotate(n_lay, (trainable_parameters_list[iii] - 8e3, accuracy_val_list[iii] + 1.2e-3), fontsize=22)
    plt.ylim([0.97, 1.005])
    plt.xlim([4.5e5, 1.14e6])
    name_fig = './plots/hyperparam_selection_layers.pdf'
    plt.savefig(name_fig)
    plt.close()

    #######################################
    # PLOTS VARYING THE NUMBER OF FILTER #
    #######################################

    models_params = ['256,256,256,256,256-7,7,7,5,3',
                     '128,128,128,128,128-7,7,7,5,3',
                     '64,64,64,64,64-7,7,7,5,3',
                     '32,32,32,32,32-7,7,7,5,3',
                     '16,16,16,16,16-7,7,7,5,3']

    trainable_parameters_list = []
    accuracy_val_list = []

    for mod_idx in range(len(models_params)):
        model_n_params = models_params[mod_idx]
        model_n = './outputs/' + args.model_name + model_n_params + '_IDrecs[\'57\']_TX[0, 1, 2]_RX[0]_' \
                  'posTRAIN[1, 2, 3, 4, 5, 6, 7, 8, 9]_' \
                  'posTEST[1, 2, 3, 4, 5, 6, 7, 8, 9]_bandwidth80_' \
                  'MOD' + args.model_type + '_hyper_selection-' + model_n_params + '.txt'

        with open(model_n, "rb") as fp:
            metrics_dict = pickle.load(fp)

        trainable_parameters_list.append(metrics_dict['trainable_parameters'])
        accuracy_val_list.append(metrics_dict['accuracy_val'])

    num_layers = [256, 128, 64, 32, 16]
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(6, 5)
    plt.plot(trainable_parameters_list, accuracy_val_list, marker='o', markersize=16)
    plt.grid(axis='y')
    plt.grid(axis='x')
    plt.xlabel(r'trainable params', fontsize=30)
    plt.ylabel(r'Accuracy [\%]', fontsize=30)
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
    for iii, n_lay in enumerate(num_layers):
        plt.annotate(n_lay, (trainable_parameters_list[iii] - 5.3e4, accuracy_val_list[iii] - 8e-3), fontsize=24)
    plt.ylim([0.92, 1])
    plt.xlim([-1e5, 2e6])
    plt.xticks([0, 5e5, 1e6, 1.5e6, 2e6])
    name_fig = './plots/hyperparam_selection_filters.pdf'
    plt.savefig(name_fig)
    plt.close()
