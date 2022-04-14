
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
from dataset_utility import *
from plots_utility import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_name', help='Model name')
    parser.add_argument('receivers', help='Receivers ids')
    parser.add_argument('positions', help='Number of different positions', type=int)
    parser.add_argument('M', help='Number of transmitting antennas', type=int)
    parser.add_argument('N', help='Number of receiving antennas', type=int)
    parser.add_argument('tx_antennas', help='Indices of TX antennas to consider (comma separated)')
    parser.add_argument('rx_antennas', help='Indices of RX antennas to consider (comma separated)')
    parser.add_argument('bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                          '(default 80)', type=int)
    parser.add_argument('model_type', help='convolutional or attention')
    parser.add_argument('scenario', help='Scenario considered, in {S1, S2, S3, S4, S4_diff, S5, S6, hyper}')
    args = parser.parse_args()

    M = args.M
    N = args.N
    positions = args.positions

    bandwidth = args.bandwidth

    module_IDs = ['49', '4F', '50', '51', '99', '9A', '9B', '9D', 'A3', 'A4']

    scenario = args.scenario
    if scenario == 'S1':
        # S1
        pos_train_val = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        pos_test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif scenario == 'S2':
        # S2
        pos_train_val = [1, 3, 5, 7, 9]
        pos_test = [2, 4, 6, 8]
    elif scenario == 'S3':
        # S3
        pos_train_val = [1, 2, 3, 4, 5]
        pos_test = [6, 7, 8, 9]
    elif scenario == 'S4':
        # S4 mobility
        pos_train_val = [5, 6, 7, 8]
        pos_test = [9, 10, 11]
    elif scenario == 'S4_diff':
        # S4 different mobility
        pos_train_val = [5, 6, 7, 8]
        pos_test = [9, 10, 11]
    elif scenario == 'S5':
        # S5 mobility
        pos_train_val = [1, 2, 3, 4]
        pos_test = [5, 6, 7, 8, 9, 10, 11]
    elif scenario == 'S6':
        # S6 mobility
        pos_train_val = [5, 6, 7, 8, 9, 10, 11]
        pos_test = [1, 2, 3, 4]

    tx_antennas = args.tx_antennas
    tx_antennas_list = []
    for lab_act in tx_antennas.split(','):
        lab_act = int(lab_act)
        if lab_act >= M:
            print('error in the tx_antennas input arg')
            break
        tx_antennas_list.append(lab_act)

    rx_antennas = args.rx_antennas
    rx_antennas_list = []
    for lab_act in rx_antennas.split(','):
        lab_act = int(lab_act)
        if lab_act >= N:
            print('error in the rx_antennas input arg')
            break
        rx_antennas_list.append(lab_act)

    num_pos = args.positions

    # RECEIVERs
    receivers = args.receivers
    receivers_list = []
    for rec in receivers.split(','):
        receivers_list.append(rec)

    model_name = args.model_name
    name_save = model_name + \
                'IDrecs' + str(receivers_list) + \
                '_TX' + str(tx_antennas_list) + \
                '_RX' + str(rx_antennas_list) + \
                '_posTRAIN' + str(pos_train_val) + \
                '_posTEST' + str(pos_test) + \
                '_bandwidth' + str(bandwidth) + \
                '_MOD' + args.model_type

    name_file = './outputs/' + name_save + '.txt'

    with open(name_file, "rb") as fp:
        metrics_dict = pickle.load(fp)

    conf_matrix_train = metrics_dict['conf_matrix_train']
    accuracy_train = metrics_dict['accuracy_train']
    precision_train = metrics_dict['precision_train']
    recall_train = metrics_dict['recall_train']
    fscore_train = metrics_dict['fscore_train']

    name_plot = name_save + '_TRAIN'
    plt_confusion_matrix(len(module_IDs), conf_matrix_train, activities=module_IDs, name=name_plot)

    conf_matrix_val = metrics_dict['conf_matrix_val']
    accuracy_val = metrics_dict['accuracy_val']
    precision_val = metrics_dict['precision_val']
    recall_val = metrics_dict['recall_val']
    fscore_val = metrics_dict['fscore_val']

    name_plot = name_save + '_VAL'
    plt_confusion_matrix(len(module_IDs), conf_matrix_val, activities=module_IDs, name=name_plot)

    conf_matrix_test = metrics_dict['conf_matrix_test']
    accuracy_test = metrics_dict['accuracy_test']
    precision_test = metrics_dict['precision_test']
    recall_test = metrics_dict['recall_test']
    fscore_test = metrics_dict['fscore_test']

    name_plot = name_save + '_TEST'
    plt_confusion_matrix(len(module_IDs), conf_matrix_test, activities=module_IDs, name=name_plot)

    string_latex = ''
    for row in range(len(module_IDs)):
        for col in range(len(module_IDs)):
            string_latex = string_latex + '(' + str(row) + ',' + str(col) + ') [' \
                           + str(conf_matrix_test[row, col]) + '] '
        string_latex = string_latex + '\n\n'
