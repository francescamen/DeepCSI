
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

import tensorflow as tf
import scipy.io as sio
import numpy as np
import pickle


def load_numpy(name_f, start_fraction, end_fraction, selected_subcarriers_idxs=None):
    v_matrices = None
    start_p = 0
    end_p = 0
    if name_f.endswith('mat'):
        v_matrices = sio.loadmat(name_f)
        v_matrices = (v_matrices['vtilde_matrices'])
        num_samples = v_matrices.shape[1]

        start_p = int(start_fraction * num_samples)
        end_p = int(end_fraction * num_samples)

        v_matrices = v_matrices[0, start_p:end_p]
        v_matrices = np.stack(v_matrices, axis=0)

    elif name_f.endswith('txt'):
        with open(name_f, "rb") as fp:
            v_matrices = pickle.load(fp)
        num_samples = v_matrices.shape[0]

        start_p = int(start_fraction * num_samples)
        end_p = int(end_fraction * num_samples)
        v_matrices = v_matrices[start_p:end_p, ...]

    if selected_subcarriers_idxs is not None:
        v_matrices = v_matrices[:, :, selected_subcarriers_idxs, :]

    return v_matrices, start_p, end_p


def load_real_imag(name_f, start_fraction, end_fraction, M, tx_antennas_list, N, rx_antennas_list,
                   selected_subcarriers_idxs=None):
    v_mat_stack, start_p, end_p = load_numpy(name_f, start_fraction, end_fraction, selected_subcarriers_idxs)

    tx_antennas_del = [i for i in range(M) if i not in tx_antennas_list]
    rx_antennas_del = [i for i in range(N) if i not in rx_antennas_list]

    v_mat_stack = np.delete(v_mat_stack, tx_antennas_del, axis=3)
    v_mat_stack = np.delete(v_mat_stack, rx_antennas_del, axis=1)

    v_mat_stack_real = np.real(v_mat_stack)
    v_mat_stack_imag = np.imag(v_mat_stack)

    # remove imaginary part of last tx antenna as it is always 0
    if M-1 in tx_antennas_list:
        v_mat_stack_imag = v_mat_stack_imag[..., :-1]

    v_mat_stack = np.concatenate([v_mat_stack_real, v_mat_stack_imag], axis=3)
    return v_mat_stack, start_p, end_p


def load_data(name_f, start_fraction, end_fraction, label, M, tx_antennas_list, N, rx_antennas_list,
              selected_subcarriers_idxs=None):
    v_mat_stack, start_p, end_p = load_real_imag(name_f, start_fraction, end_fraction,
                                                 M, tx_antennas_list, N, rx_antennas_list, selected_subcarriers_idxs)
    v_tensor = tf.cast(v_mat_stack, tf.float32)
    labels_vector = label * tf.ones((end_p - start_p), dtype=tf.dtypes.int32)
    return v_tensor, labels_vector


def create_dataset(name_files, labels, batch_size, M, tx_antennas_list, N, rx_antennas_list, shuffle, cache_file,
                   prefetch=True, repeat=True, start_fraction=0, end_fraction=0.8, selected_subcarriers_idxs=None):
    labels_complete = []

    v_tensor, labels_vector = load_data(name_files[0], start_fraction, end_fraction, labels[0],
                                        M, tx_antennas_list, N, rx_antennas_list, selected_subcarriers_idxs)
    dataset_complete = tf.data.Dataset.from_tensor_slices((v_tensor, labels_vector))

    length_dataset = len(labels_vector)
    labels_complete.extend(labels_vector)

    for idx in range(1, len(labels)):
        v_tensor, labels_vector = load_data(name_files[idx], start_fraction, end_fraction, labels[idx],
                                            M, tx_antennas_list, N, rx_antennas_list, selected_subcarriers_idxs)
        dataset_file = tf.data.Dataset.from_tensor_slices((v_tensor, labels_vector))

        dataset_complete = dataset_complete.concatenate(dataset_file)
        length_dataset = length_dataset + len(labels_vector)
        labels_complete.extend(labels_vector)

    dataset_complete = dataset_complete.cache(cache_file)

    if shuffle:
        dataset_complete = dataset_complete.shuffle(length_dataset)
    if repeat:
        dataset_complete = dataset_complete.repeat()
    dataset_complete = dataset_complete.batch(batch_size=batch_size)
    if prefetch:
        dataset_complete = dataset_complete.prefetch(buffer_size=1)
    return dataset_complete, length_dataset, labels_complete
