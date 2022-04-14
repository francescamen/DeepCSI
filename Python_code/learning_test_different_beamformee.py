
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
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from network_utility import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('receivers', help='Receivers ids, first for training, second for testing')
    parser.add_argument('positions', help='Number of different positions', type=int)
    parser.add_argument('model_name', help='Model name')
    parser.add_argument('M', help='Number of transmitting antennas', type=int)
    parser.add_argument('N', help='Number of receiving antennas', type=int)
    parser.add_argument('tx_antennas', help='Indices of TX antennas to consider (comma separated)')
    parser.add_argument('rx_antennas', help='Indices of RX antennas to consider (comma separated)')
    parser.add_argument('bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                          '(default 80)', type=int)
    parser.add_argument('model_type', help='convolutional or attention')
    parser.add_argument('prefix', help='Prefix')
    parser.add_argument('scenario', help='Scenario considered, in {S1, S2, S3, S4, S4_diff, S5, S6, hyper}')
    args = parser.parse_args()

    prefix = args.prefix
    model_name = args.model_name
    if os.path.exists('./cache_files/' + model_name + '_cache_test.data-00000-of-00001'):
        os.remove('./cache_files/' + model_name + '_cache_test.data-00000-of-00001')
        os.remove('./cache_files/' + model_name + '_cache_test.index')

    scenario = args.scenario
    if scenario == 'S1':
        # S1
        pos_train_val = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        pos_test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_fraction = [0.8, 1]
    elif scenario == 'S2':
        # S2
        pos_train_val = [1, 3, 5, 7, 9]
        pos_test = [2, 4, 6, 8]
        test_fraction = [0, 1]
    elif scenario == 'S3':
        # S3
        pos_train_val = [1, 2, 3, 4, 5]
        pos_test = [6, 7, 8, 9]
        test_fraction = [0, 1]

    # RECEIVERs train and test
    receivers = args.receivers
    receivers_list = []
    for rec in receivers.split(','):
        receivers_list.append(rec)
    train_RX = receivers_list[0]
    test_RX = receivers_list[1]

    # TX and RX antennas selection
    M = args.M
    N = args.N

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

    # Subcarriers selection
    selected_subcarriers_idxs = None  # default, i.e., 80 MHz
    num_selected_subcarriers = 234
    bandwidth = args.bandwidth
    if bandwidth == 40:
        subc_block1 = np.arange(0, 55, 1)  # remove 2 for the pilot sub-channels in position -11 and -39 not sounded
        subc_block2 = np.arange(58, 113, 1)  # remove 2 for the pilot sub-channels in position -75 and -103 not sounded
        selected_subcarriers_idxs = np.concatenate((subc_block1, subc_block2))
        num_selected_subcarriers = selected_subcarriers_idxs.shape[0]
    elif bandwidth == 20:
        subc_block1 = np.arange(0, 27, 1)  # remove 1 for the pilot sub-channels in position -11
        subc_block2 = np.arange(28, 55, 1)  # remove 1 for the pilot sub-channels in position -39
        selected_subcarriers_idxs = np.concatenate((subc_block1, subc_block2))
        num_selected_subcarriers = selected_subcarriers_idxs.shape[0]

    # LOAD TEST DATA OF test_RX
    input_dir = args.dir + test_RX + '/'
    num_pos = args.positions
    extension = '.txt'

    module_IDs = ['49', '4F', '50', '51', '99', '9A', '9B', '9D', 'A3', 'A4']

    name_files_test = []
    labels_test = []
    for mod_label, mod_ID in enumerate(module_IDs):
        for pos in pos_test:
            name_file = input_dir + mod_ID + prefix + str(pos - 1) + extension
            name_files_test.append(name_file)
            labels_test.append(mod_label)

    batch_size = 32
    name_cache_test = './cache_files/' + model_name + '_cache_test'
    dataset_test, num_samples_test, labels_complete_test = create_dataset(name_files_test, labels_test, batch_size,
                                                                          M, tx_antennas_list, N, rx_antennas_list,
                                                                          shuffle=False, cache_file=name_cache_test,
                                                                          prefetch=True, repeat=True,
                                                                          start_fraction=test_fraction[0],
                                                                          end_fraction=test_fraction[1],
                                                                          selected_subcarriers_idxs
                                                                          =selected_subcarriers_idxs)

    IQ_dimension = 2
    N_considered = len(rx_antennas_list)
    M_considered = len(tx_antennas_list)
    input_shape = (N_considered, num_selected_subcarriers, M_considered * IQ_dimension)
    if M - 1 in tx_antennas_list:
        # -1 because last tx antenna has only real part
        input_shape = (N_considered, num_selected_subcarriers, M_considered * IQ_dimension - 1)
    print(input_shape)

    num_classes = len(module_IDs)

    # LOAD MODEL OF train_RX
    model_type = args.model_type
    name_load = model_name + \
                'IDrecs' + str([train_RX]) + \
                '_TX' + str(tx_antennas_list) + \
                '_RX' + str(rx_antennas_list) + \
                '_posTRAIN' + str(pos_train_val) + \
                '_posTEST' + str(pos_test) + \
                '_bandwidth' + str(bandwidth) + \
                '_MOD' + args.model_type
    name_model = './network_models/' + name_load + 'network.h5'

    custom_objects = {'ConvNormalization': ConvNormalization}

    model_net = tf.keras.models.load_model(name_model, custom_objects=custom_objects)

    # TEST
    test_steps_per_epoch = int(np.ceil(num_samples_test / batch_size))
    prediction_test = model_net.predict(dataset_test, steps=test_steps_per_epoch)[:len(labels_complete_test)]

    labels_pred_test = np.argmax(prediction_test, axis=1)

    labels_complete_test_array = np.asarray(labels_complete_test)
    conf_matrix_test = confusion_matrix(labels_complete_test_array, labels_pred_test,
                                        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                        normalize='true')
    precision_test, recall_test, fscore_test, _ = precision_recall_fscore_support(labels_complete_test_array,
                                                                                  labels_pred_test,
                                                                                  labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    accuracy_test = accuracy_score(labels_complete_test_array, labels_pred_test)
    print('Accuracy test: %.5f' % accuracy_test)

    metrics_dict = {'conf_matrix_test': conf_matrix_test, 'accuracy_test': accuracy_test,
                    'precision_test': precision_test, 'recall_test': recall_test, 'fscore_test': fscore_test
                    }

    name_save = model_name + \
                'IDrec_train' + str([train_RX]) + \
                'IDrec_test' + test_RX + \
                '_TX' + str(tx_antennas_list) + \
                '_RX' + str(rx_antennas_list) + \
                '_posTRAIN' + str(pos_train_val) + \
                '_posTEST' + str(pos_test) + \
                '_bandwidth' + str(bandwidth) + \
                '_MOD' + args.model_type
    name_file = './outputs/' + name_save + '.txt'

    with open(name_file, "wb") as fp:
        pickle.dump(metrics_dict, fp)

    string_latex = ''
    for row in range(len(module_IDs)):
        for col in range(len(module_IDs)):
            string_latex = string_latex + '(' + str(row) + ',' + str(col) + ') [' + str(conf_matrix_test[row, col]) + '] '
        string_latex = string_latex + '\n\n'
