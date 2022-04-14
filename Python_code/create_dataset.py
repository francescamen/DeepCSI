
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
import pickle
from dataset_utility import load_numpy
import numpy as np
import math as mt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('receiver', help='Receiver id')
    parser.add_argument('positions', help='Number of different positions', type=int)
    parser.add_argument('num_samples', help='Max number of samples to use', type=int)
    parser.add_argument('prefix', help='Prefix')
    parser.add_argument('save_folder', help='Folder where to save the dataset')
    parser.add_argument('rand', help='Select random indices or sub-sample the trace: type `random` or `sampling`')
    args = parser.parse_args()

    input_dir = args.dir + args.receiver + '/vtilde_matrices/'
    num_pos = args.positions
    max_num_samples = args.num_samples
    prefix = args.prefix
    save_folder = args.save_folder

    module_IDs = ['49', '4F', '50', '51', '99', '9A', '9B', '9D', 'A3', 'A4']

    for idx in range(len(module_IDs)):
        for pos in range(num_pos):
            if pos == 10:
                pos = 'A'
            name_file = module_IDs[idx] + prefix + str(pos)
            name_v_mat = input_dir + name_file + '.mat'
            try:
                v_matrix, _, _ = load_numpy(name_v_mat, 0, 1)
            except FileNotFoundError:
                print('file ', name_v_mat, ' doesn\'t exist')
                continue

            # Select samples to balance the dataset
            num_sampl = v_matrix.shape[0]
            if num_sampl > max_num_samples:
                if args.rand == 'random':
                    selected_idxs = np.random.randint(0, num_sampl, max_num_samples)
                    selected_idxs = np.sort(selected_idxs)
                elif args.rand == 'sampling':
                    selected_idxs = np.arange(0, num_sampl, mt.ceil(num_sampl/max_num_samples))
                v_matrix = v_matrix[selected_idxs, ...]

            name_file = save_folder + args.receiver + '/' + name_file + '.txt'
            with open(name_file, "wb") as fp:
                pickle.dump(v_matrix, fp)
