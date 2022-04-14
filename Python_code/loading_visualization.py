
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
import scipy.io as sio
from dataset_utility import load_real_imag
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times'
rcParams['text.usetex'] = 'true'
rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
rcParams['font.size'] = 16

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('receiver', help='Receiver id')
    parser.add_argument('positions', help='Number of different positions', type=int)
    parser.add_argument('M', help='Number of transmitting antennas', type=int)
    parser.add_argument('N', help='Number of receiving antennas', type=int)
    parser.add_argument('prefix', help='Prefix')
    parser.add_argument('save_folder', help='Folder where to save the dataset')
    args = parser.parse_args()

    input_dir = args.dir + args.receiver + '/vtilde_matrices/'
    input_dir_time = args.dir + args.receiver + '/time_vector/'
    num_pos = args.positions
    M = args.M
    N = args.N
    prefix = args.prefix
    save_folder = args.save_folder

    module_IDs = ['49', '4F', '50', '51', '99', '9A', '9B', '9D', 'A3', 'A4']
    tx_antennas_list = [i for i in range(M)]
    rx_antennas_list = [i for i in range(N)]

    for pos in range(num_pos):
        for idx in range(len(module_IDs)):
            if pos == 10:
                pos = 'A'
            name_file = module_IDs[idx] + prefix + str(pos) + '.mat'

            name_v_mat = input_dir + name_file
            try:
                v_matrix, _, _ = load_real_imag(name_v_mat, 0, 1, M, tx_antennas_list, N, rx_antennas_list)
            except FileNotFoundError:
                print('file ', name_v_mat, ' doesn\'t exist')
                continue
            v_matrix = v_matrix / np.mean(np.abs(v_matrix), axis=2, keepdims=True)

            name_time_vect = input_dir_time + name_file
            time_vector = sio.loadmat(name_time_vect)
            time_vector = np.concatenate((time_vector['time_vector'])[0, :])
            time_vector = time_vector/1E6
            time_vector = time_vector - time_vector[0]

            # TIME VECTOR
            plt.figure()
            plt.stem(time_vector)
            plt.xlabel(r'sample idx')
            plt.ylabel(r'time [s]')
            name_plot = save_folder + args.receiver + '/time/' + str(module_IDs[idx]) + '_pos' + prefix + str(pos) + \
                        '_time'
            plt.savefig(name_plot, bbox_inches='tight')
            plt.close()

            # REAL PART
            fig = plt.figure(constrained_layout=True)
            fig.set_size_inches(15, 9)
            widths = [1, 1, 1, 0.5]
            heights = [1, 1]
            gs = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths, height_ratios=heights)

            ax = []
            plt1 = None
            for stream in range(N):
                for ant in range(M):
                    ax1 = fig.add_subplot(gs[(stream, ant)])

                    plt1 = ax1.pcolormesh(v_matrix[:, stream, :, ant].T, cmap='viridis', linewidth=0, rasterized=True)
                    plt1.set_edgecolor('face')
                    ax1.set_ylabel(r'subcarrier idx')
                    ax1.set_xlabel(r'time idx')

                    title_p = 'tx ant: ' + str(ant) + ' - stream: ' + str(stream)
                    ax1.set_title(title_p)
                    ax.append(ax1)

            cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
            cbar1 = fig.colorbar(plt1, cax=cbar_ax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
            cbar1.ax.set_ylabel('real part')

            plt.suptitle('transmitter:  ' + str(module_IDs[idx]), fontsize=15)
            for axi in ax:
                axi.label_outer()

            name_plot = save_folder + args.receiver + '/' + str(module_IDs[idx]) + '_pos' + prefix + str(pos) + '_real'
            plt.savefig(name_plot, bbox_inches='tight')
            plt.close()

            # IMAGINARY PART
            fig = plt.figure(constrained_layout=True)
            fig.set_size_inches(15, 9)
            widths = [1, 1, 1, 0.5]
            heights = [1, 1]
            gs = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths, height_ratios=heights)

            ax = []
            for stream in range(N):
                for ant in range(M - 1):
                    ax1 = fig.add_subplot(gs[(stream, ant)])

                    plt1 = ax1.pcolormesh(v_matrix[:, stream, :, ant + 3].T, cmap='viridis', linewidth=0,
                                          rasterized=True)
                    plt1.set_edgecolor('face')
                    ax1.set_ylabel(r'subcarrier idx')
                    ax1.set_xlabel(r'time idx')

                    title_p = 'tx ant: ' + str(ant) + ' - stream: ' + str(stream)
                    ax1.set_title(title_p)
                    ax.append(ax1)

            cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
            cbar1 = fig.colorbar(plt1, cax=cbar_ax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
            cbar1.ax.set_ylabel('imaginary part')

            plt.suptitle('transmitter:  ' + str(module_IDs[idx]), fontsize=15)
            for axi in ax:
                axi.label_outer()

            name_plot = save_folder + args.receiver + '/' + str(module_IDs[idx]) + '_pos' + prefix + str(pos) + '_imag'
            plt.savefig(name_plot, bbox_inches='tight')
            plt.close()
