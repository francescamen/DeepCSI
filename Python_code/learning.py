
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
import shutil
from network_utility import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('receivers', help='Receivers ids')
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
    list_cache_files = os.listdir('./cache_files/')
    for file_cache in list_cache_files:
        if file_cache.startswith(model_name):
            os.remove('./cache_files/' + file_cache)
    if os.path.exists('./logs/train/'):
        shutil.rmtree('./logs/train/')
    if os.path.exists('./logs/validation/'):
        shutil.rmtree('./logs/validation/')

    scenario = args.scenario
    if scenario == 'S1':
        # S1
        pos_train_val = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_fraction = [0, 0.64]
        val_fraction = [0.64, 0.8]
        pos_test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_fraction = [0.8, 1]
    elif scenario == 'S2':
        # S2
        pos_train_val = [1, 3, 5, 7, 9]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [2, 4, 6, 8]
        test_fraction = [0, 1]
    elif scenario == 'S3':
        # S3
        pos_train_val = [1, 2, 3, 4, 5]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [6, 7, 8, 9]
        test_fraction = [0, 1]
    elif scenario == 'S4':
        # S4 mobility
        pos_train_val = [5, 6, 7, 8]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [9, 10, 11]
        test_fraction = [0, 1]
    elif scenario == 'S4_diff':
        # S4 different mobility
        pos_train_val = [5, 6, 7, 8]
        train_fraction = [0, 0.4]
        val_fraction = [0.4, 0.5]
        pos_test = [9, 10, 11]
        test_fraction = [0.6, 0.8]
    elif scenario == 'S5':
        # S5 mobility
        pos_train_val = [1, 2, 3, 4]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [5, 6, 7, 8, 9, 10, 11]
        test_fraction = [0, 1]
    elif scenario == 'S6':
        # S6 mobility
        pos_train_val = [5, 6, 7, 8, 9, 10, 11]
        train_fraction = [0, 0.8]
        val_fraction = [0.8, 1]
        pos_test = [1, 2, 3, 4]
        test_fraction = [0, 1]
    elif scenario == 'hyper':
        # Hyper parameters selection
        pos_train_val = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_fraction = [0, 0.5]
        val_fraction = [0.5, 0.8]
        pos_test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_fraction = [0.8, 1]

    # Positions and device IDs
    num_pos = args.positions
    extension = '.txt'
    module_IDs = ['49', '4F', '50', '51', '99', '9A', '9B', '9D', 'A3', 'A4']

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

    # RECEIVERs
    receivers = args.receivers
    receivers_list = []
    for rec in receivers.split(','):
        receivers_list.append(rec)

    name_files_train = []
    labels_train = []
    name_files_val = []
    labels_val = []
    name_files_test = []
    labels_test = []

    for rec in receivers_list:
        input_dir = args.dir + rec + '/'

        for mod_label, mod_ID in enumerate(module_IDs):
            for pos in pos_train_val:
                pos_id = pos - 1
                if pos_id == 10:
                    pos_id = 'A'
                name_file = input_dir + mod_ID + prefix + str(pos_id) + extension
                name_files_train.append(name_file)
                labels_train.append(mod_label)

        for mod_label, mod_ID in enumerate(module_IDs):
            for pos in pos_train_val:
                pos_id = pos - 1
                if pos_id == 10:
                    pos_id = 'A'
                name_file = input_dir + mod_ID + prefix + str(pos_id) + extension
                name_files_val.append(name_file)
                labels_val.append(mod_label)

        for mod_label, mod_ID in enumerate(module_IDs):
            for pos in pos_test:
                pos_id = pos - 1
                if pos_id == 10:
                    pos_id = 'A'
                name_file = input_dir + mod_ID + prefix + str(pos_id) + extension
                name_files_test.append(name_file)
                labels_test.append(mod_label)

    batch_size = 32
    name_cache_train = './cache_files/' + model_name + 'cache_train'
    dataset_train, num_samples_train, labels_complete_train = create_dataset(name_files_train, labels_train, batch_size,
                                                                             M, tx_antennas_list, N, rx_antennas_list,
                                                                             shuffle=True, cache_file=name_cache_train,
                                                                             prefetch=True, repeat=True,
                                                                             start_fraction=train_fraction[0],
                                                                             end_fraction=train_fraction[1],
                                                                             selected_subcarriers_idxs=
                                                                             selected_subcarriers_idxs)

    name_cache_val = './cache_files/' + model_name + 'cache_val'
    dataset_val, num_samples_val, labels_complete_val = create_dataset(name_files_val, labels_val, batch_size,
                                                                       M, tx_antennas_list, N, rx_antennas_list,
                                                                       shuffle=False, cache_file=name_cache_val,
                                                                       prefetch=True, repeat=True,
                                                                       start_fraction=val_fraction[0],
                                                                       end_fraction=val_fraction[1],
                                                                       selected_subcarriers_idxs=
                                                                       selected_subcarriers_idxs)

    name_cache_test = './cache_files/' + model_name + 'cache_test'
    dataset_test, num_samples_test, labels_complete_test = create_dataset(name_files_test, labels_test, batch_size,
                                                                          M, tx_antennas_list, N, rx_antennas_list,
                                                                          shuffle=False, cache_file=name_cache_test,
                                                                          prefetch=True, repeat=True,
                                                                          start_fraction=test_fraction[0],
                                                                          end_fraction=test_fraction[1],
                                                                          selected_subcarriers_idxs=
                                                                          selected_subcarriers_idxs)

    IQ_dimension = 2
    N_considered = len(rx_antennas_list)
    M_considered = len(tx_antennas_list)
    input_shape = (N_considered, num_selected_subcarriers, M_considered * IQ_dimension)
    if M - 1 in tx_antennas_list:
        # -1 because last tx antenna has only real part
        input_shape = (N_considered, num_selected_subcarriers, M_considered * IQ_dimension - 1)
    print(input_shape)

    num_classes = len(module_IDs)

    # MODEL and OPTIMIZER
    model_type = args.model_type
    model_net = None
    optimiz = None
    if model_type == 'convolutional':
        model_net = conv_network(input_shape, num_classes, model_name)
        optimiz = tf.keras.optimizers.Adam(learning_rate=5E-5)
    elif model_type[:29] == 'convolutional_hyper_selection':
        hyper_parameters = model_type[30:]
        hyper_parameters_list = [hp for hp in hyper_parameters.split('-')]
        filters_dimension = [int(filt) for filt in hyper_parameters_list[0].split(',')]
        kernels_dimension = [int(kern) for kern in hyper_parameters_list[1].split(',')]
        model_net = conv_network_hyper_selection(input_shape, num_classes, filters_dimension,
                                                 kernels_dimension, model_name)
        model_name = model_name + hyper_parameters + '_'
        optimiz = tf.keras.optimizers.Adam(learning_rate=5E-5)
    elif model_type == 'attention':
        model_net = att_network(input_shape, num_classes)
        optimiz = tf.keras.optimizers.Adam(learning_rate=3E-5)
    elif model_type[:25] == 'attention_hyper_selection':
        hyper_parameters = model_type[26:]
        hyper_parameters_list = [hp for hp in hyper_parameters.split('-')]
        filters_dimension = [int(filt) for filt in hyper_parameters_list[0].split(',')]
        kernels_dimension = [int(kern) for kern in hyper_parameters_list[1].split(',')]
        model_net = att_network_hyper_selection(input_shape, num_classes, filters_dimension,
                                                kernels_dimension, model_name)
        model_name = model_name + hyper_parameters + '_'
        optimiz = tf.keras.optimizers.Adam(learning_rate=3E-5)
    else:
        print('Allowed values for the model_type argument are: convolutional, inception, dense, attention, '
              'convolutional_hyper_selection')

    model_net.summary()

    # TRAIN
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='True')
    model_net.compile(optimizer=optimiz, loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    train_steps_per_epoch = int(np.ceil(num_samples_train / batch_size))
    val_steps_per_epoch = int(np.ceil(num_samples_val / batch_size))
    test_steps_per_epoch = int(np.ceil(num_samples_test / batch_size))

    callback_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    name_save = model_name + \
                'IDrecs' + str(receivers_list) + \
                '_TX' + str(tx_antennas_list) + \
                '_RX' + str(rx_antennas_list) + \
                '_posTRAIN' + str(pos_train_val) + \
                '_posTEST' + str(pos_test) + \
                '_bandwidth' + str(bandwidth) + \
                '_MOD' + args.model_type

    name_model = './network_models/' + name_save + 'network.h5'

    callback_save = tf.keras.callbacks.ModelCheckpoint(name_model, save_freq='epoch', save_best_only=True,
                                                       monitor='val_sparse_categorical_accuracy')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    results = model_net.fit(dataset_train, epochs=30, steps_per_epoch=train_steps_per_epoch,
                            validation_data=dataset_val, validation_steps=val_steps_per_epoch,
                            callbacks=[callback_save, tensorboard_callback])

    custom_objects = {'ConvNormalization': ConvNormalization}

    best_model = tf.keras.models.load_model(name_model, custom_objects=custom_objects)
    model_net = best_model

    # TEST
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

    # VAL
    prediction_val = model_net.predict(dataset_val, steps=val_steps_per_epoch)[:len(labels_complete_val)]

    labels_pred_val = np.argmax(prediction_val, axis=1)

    labels_complete_val_array = np.asarray(labels_complete_val)
    conf_matrix_val = confusion_matrix(labels_complete_val_array, labels_pred_val,
                                       labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                       normalize='true')
    precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(labels_complete_val_array,
                                                                               labels_pred_val,
                                                                               labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    accuracy_val = accuracy_score(labels_complete_val_array, labels_pred_val)
    print('Accuracy val: %.5f' % accuracy_val)

    # TRAIN TEST
    name_cache_train_test = './cache_files/' + model_name + 'cache_train_test'
    dataset_train, num_samples_train, labels_complete_train = create_dataset(name_files_train, labels_train, batch_size,
                                                                             M, tx_antennas_list, N, rx_antennas_list,
                                                                             shuffle=False,
                                                                             cache_file=name_cache_train_test,
                                                                             prefetch=True, repeat=True,
                                                                             start_fraction=train_fraction[0],
                                                                             end_fraction=train_fraction[1],
                                                                             selected_subcarriers_idxs=
                                                                             selected_subcarriers_idxs)

    prediction_train = model_net.predict(dataset_train, steps=train_steps_per_epoch)[:len(labels_complete_train)]

    labels_pred_train = np.argmax(prediction_train, axis=1)

    labels_complete_train_array = np.asarray(labels_complete_train)
    conf_matrix_train_test = confusion_matrix(labels_complete_train_array, labels_pred_train,
                                              labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              normalize='true')
    precision_train_test, recall_train_test, fscore_train_test, _ = precision_recall_fscore_support(
        labels_complete_train_array, labels_pred_train, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    accuracy_train_test = accuracy_score(labels_complete_train_array, labels_pred_train)
    print('Accuracy train: %.5f' % accuracy_train_test)

    trainable_parameters = np.sum([np.prod(v.get_shape()) for v in model_net.trainable_weights])
    metrics_dict = {'trainable_parameters': trainable_parameters,
                    'conf_matrix_train': conf_matrix_train_test, 'accuracy_train': accuracy_train_test,
                    'precision_train': precision_train_test, 'recall_train': recall_train_test,
                    'fscore_train': fscore_train_test,
                    'conf_matrix_val': conf_matrix_val, 'accuracy_val': accuracy_val,
                    'precision_val': precision_val, 'recall_val': recall_val, 'fscore_val': fscore_val,
                    'conf_matrix_test': conf_matrix_test, 'accuracy_test': accuracy_test,
                    'precision_test': precision_test, 'recall_test': recall_test, 'fscore_test': fscore_test
                    }

    name_file = './outputs/' + name_save + '.txt'

    with open(name_file, "wb") as fp:
        pickle.dump(metrics_dict, fp)
