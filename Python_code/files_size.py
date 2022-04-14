
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
import os
import numpy as np
from dataset_utility import load_numpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('receiver', help='Receiver id')
    parser.add_argument('positions', help='Number of different positions', type=int)
    parser.add_argument('prefix', help='Prefix')
    args = parser.parse_args()

    input_dir = args.dir + args.receiver + '/vtilde_matrices/'
    input_dir_time = args.dir + args.receiver + '/time_vector/'
    num_pos = args.positions
    prefix = args.prefix

    module_IDs = ['49', '4F', '50', '51', '99', '9A', '9B', '9D', 'A3', 'A4']

    name_files = []
    labels = []
    for mod_label, mod_ID in enumerate(module_IDs):
        for pos in range(num_pos):
            if pos == 10:
                pos = 'A'
            name_file = mod_ID + prefix + str(pos) + '.mat'
            name_files.append(name_file)
            labels.append(mod_label)

    num_IDs = len(module_IDs)
    files_size = np.nan*np.zeros((num_IDs, num_pos))
    num_samples = np.nan*np.zeros((num_IDs, num_pos), dtype=int)
    idxs = np.arange(0, num_pos*num_IDs, num_pos)
    for ID in range(num_IDs):
        mod_id = module_IDs[ID]
        for pos in range(num_pos):
            idx = idxs[ID] + pos
            name_v_mat = input_dir + name_files[idx]
            try:
                file_size = os.path.getsize(name_v_mat)
            except FileNotFoundError:
                continue
            files_size[ID, pos] = file_size

            v_matrix, _, _ = load_numpy(name_v_mat, 0, 1)
            num_samples[ID, pos] = v_matrix.shape[0]
    min_num_samples = np.nanmin(num_samples)
    max_num_samples = np.nanmax(num_samples)
