
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
from dataset_utility import load_numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
import scipy.stats as st
import matplotlib as mpl
import pickle
import h5py
import matplotlib.gridspec as gridspec


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Palatino'
mpl.rcParams['text.usetex'] = 'true'
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set3.colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('receiver', help='Receiver id')
    args = parser.parse_args()

    input_dir = args.dir + args.receiver + '/vtilde_matrices/'

    num_subcarriers = 234

    #################################
    # PLOTS PDF QUANTIZATION ERRORS #
    #################################
    mpl.rcParams['font.size'] = 22
    psi_bit = 5
    name_save_v = '../Matlab_code/simulation_outputs/v_simulation_psi' + str(psi_bit) + '.mat'
    v_simulation = {}
    f = h5py.File(name_save_v, 'r')
    for k, v in f.items():
        v_simulation[k] = np.array(v)
    # v_simulation = sio.loadmat(name_save_v)
    v_simulation = (v_simulation['vs'])
    v_simulation = v_simulation['real'] + 1j * v_simulation['imag']
    v_simulation = np.moveaxis(v_simulation, [0, 1, 2, 3], [3, 2, 1, 0])

    name_save_v_rec = '../Matlab_code/simulation_outputs/v_reconstructed_simulation_psi' + str(psi_bit) + '.mat'
    v_reconstruct_simulation = {}
    f = h5py.File(name_save_v_rec, 'r')
    for k, v in f.items():
        v_reconstruct_simulation[k] = np.array(v)
    # v_reconstruct_simulation = sio.loadmat(name_save_v_rec)
    v_reconstruct_simulation = (v_reconstruct_simulation['vrs'])
    v_reconstruct_simulation = v_reconstruct_simulation['real'] + 1j * v_reconstruct_simulation['imag']
    v_reconstruct_simulation = np.moveaxis(v_reconstruct_simulation, [0, 1, 2, 3], [3, 2, 1, 0])

    error_sim = abs(v_simulation - v_reconstruct_simulation)
    error_average_subcarriers = np.mean(error_sim, axis=1)
    if psi_bit == 7:
        start_edg = 2.4e-3
        end_edg = 11.5e-3
    elif psi_bit == 5:
        start_edg = 1e-2
        end_edg = 4.2e-2
    samples = 100
    edges = np.linspace(start_edg, end_edg, samples)

    labels = [r'$[\tilde{\mathbf{V}}]_{1, 1}$', r'$[\tilde{\mathbf{V}}]_{2, 1}$',
              r'$[\tilde{\mathbf{V}}]_{3, 1}$', r'$[\tilde{\mathbf{V}}]_{1, 2}$',
              r'$[\tilde{\mathbf{V}}]_{2, 2}$', r'$[\tilde{\mathbf{V}}]_{3, 2}$']

    broken = False
    if not broken:
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(6, 5)
        samples = 500
        edges = np.linspace(start_edg, end_edg, samples)
        bin_dim = (end_edg - start_edg) / (samples - 1)
        div_factor = error_average_subcarriers.shape[1]
        for stream_n in range(2):
            for ant_n in range(3):
                dist = st.norm
                params = dist.fit(error_average_subcarriers[ant_n, :, stream_n])
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                pdf_fitted = dist.pdf(edges, loc=loc, scale=scale, *arg)/div_factor
                # plt.hist(error_average_subcarriers[ant_n, :, stream_n],
                #          label=labels[stream_n*3+ant_n], bins=edges, density=True)
                plt.plot(edges, pdf_fitted, linewidth=1, color='k', linestyle='--')
                plt.fill_between(edges, pdf_fitted, label=labels[stream_n*3+ant_n], linestyle='--', linewidth=1,
                                 edgecolor='k')

        if psi_bit == 5:
            plt.ylim([-4E-3, 6.3E-2])
        if psi_bit == 7:
            plt.ylim([-1E-2, 1.6E-1])
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        plt.grid()
        plt.legend(fontsize='medium', ncol=2, columnspacing=1)
        plt.show()
        plt.xlabel(r'$\tilde{\mathbf{V}}$ quantization error', fontsize=24)
        plt.ylabel(r'Probability density function', fontsize=24)
        name_fig = './plots/pdf_error_simulation_psi' + str(psi_bit) + '.pdf'
        plt.savefig(name_fig)
        plt.close()

    if broken:
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(6, 5)
        gs = gridspec.GridSpec(2, 1, hspace=0.15, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        samples = 500
        edges = np.linspace(start_edg, end_edg, samples)
        bin_dim = (end_edg - start_edg) / (samples - 1)
        div_factor = error_average_subcarriers.shape[1]
        for stream_n in range(2):
            for ant_n in range(3):
                dist = st.norm
                params = dist.fit(error_average_subcarriers[ant_n, :, stream_n])
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                pdf_fitted = dist.pdf(edges, loc=loc, scale=scale, *arg) / div_factor
                # plt.hist(error_average_subcarriers[ant_n, :, stream_n],
                #          label=labels[stream_n*3+ant_n], bins=edges, density=True)
                ax1.plot(edges, pdf_fitted, linewidth=1, color='k', linestyle='--')
                ax1.fill_between(edges, pdf_fitted, label=labels[stream_n * 3 + ant_n], linestyle='--', linewidth=1,
                                 edgecolor='k')
                ax2.plot(edges, pdf_fitted, linewidth=1, color='k', linestyle='--')
                ax2.fill_between(edges, pdf_fitted, label=labels[stream_n * 3 + ant_n], linestyle='--', linewidth=1,
                                 edgecolor='k')

        ax2.set_ylim([0, 0.05])
        ax1.set_ylim([0.195, 0.2])
        d = .015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax1.tick_params(labelbottom=False)
        ax1.tick_params(bottom=False)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax1.ticklabel_format(style='sci', scilimits=(-2, -2), axis='y')
        ax2.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
        ax2.ticklabel_format(style='sci', scilimits=(-2, -2), axis='y')
        ax1.grid()
        ax2.grid()
        ax1.legend(fontsize='medium', ncol=2, columnspacing=1)
        plt.show()
        ax2.set_xlabel(r'$\tilde{\mathbf{V}}$ quantization error', fontsize=24)
        fig.supylabel(r'Probability density function', fontsize=24)
        name_fig = './plots/pdf_error_simulation_psi' + str(psi_bit) + '.pdf'
        plt.savefig(name_fig)
        plt.close()

    #############################
    # PLOTS FOR HEATMAP FIGURES #
    #############################
    mpl.rcParams['font.size'] = 16
    name_file = '49_2.mat'
    name_v_mat = input_dir + name_file
    v_matrix, _, _ = load_numpy(name_v_mat, 0, 1)
    sample_start = 400  # 400
    sample_end = 430  # 430
    subcarrier_start = 0  # 92
    subcarrier_end = 76  # 168
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 5.5)
    gs = gridspec.GridSpec(2, 3, hspace=0.2, wspace=0.2, figure=fig)
    ax = []
    ticks_y = np.arange(0, subcarrier_end - subcarrier_start, 25)
    ticks_x = np.arange(0, sample_end - sample_start + 1, 15)
    # vmin = np.min(np.real(v_matrix))
    # vmax = np.max(np.real(v_matrix))
    titles = [[r'$[\tilde{\mathbf{V}}]_{1, 1}$', r'$[\tilde{\mathbf{V}}]_{2, 1}$',
               r'$[\tilde{\mathbf{V}}]_{3, 1}$'],
              [r'$[\tilde{\mathbf{V}}]_{1, 2}$', r'$[\tilde{\mathbf{V}}]_{2, 2}$',
               r'$[\tilde{\mathbf{V}}]_{3, 2}$']]
    for tx_ant in range(3):
        for rx_ant in range(2):
            v_submatrix = v_matrix[sample_start:sample_end, rx_ant, subcarrier_start:subcarrier_end, tx_ant]
            vmin = np.min(np.real(v_submatrix))
            vmax = np.max(np.real(v_submatrix))
            print('vmin %f, vmax %f' % (vmin, vmax))
            ax1 = fig.add_subplot(gs[(rx_ant, tx_ant)])
            ax1.pcolormesh(np.real(v_submatrix.T), cmap='viridis', linewidth=0, rasterized=True)  # , vmin=-1, vmax=1)
            ax1.set_title(titles[rx_ant][tx_ant])
            ax1.set_ylabel(r'sub-channel index')
            ax1.set_xlabel(r'time index')
            ax1.set_yticks(ticks_y + 0.5)
            ax1.set_yticklabels(ticks_y + subcarrier_start - 122)
            ax1.set_xticks(ticks_x)
            ax1.set_xticklabels(ticks_x)
            ax.append(ax1)
    for axi in ax:
        axi.label_outer()
    # plt.tight_layout()

    name_fig = './plots/heat_re_' + name_file[:-4] + '.pdf'
    plt.savefig(name_fig)
    plt.close()

    ##############################
    # PLOTS FOR FRAMEWORK FIGURE #
    ##############################
    mpl.rcParams['font.size'] = 18
    name_file = '49_2.mat'
    name_v_mat = input_dir + name_file
    v_matrix, _, _ = load_numpy(name_v_mat, 0, 1)
    rx_ant = 0
    sample_idx = 100
    for tx_ant in range(3):
        fig = plt.figure(constrained_layout=False)
        fig.set_size_inches(6, 3)
        plt.plot(np.real(v_matrix[sample_idx, rx_ant, :, tx_ant]), 'purple', linewidth=2.5)
        plt.grid()
        plt.ylabel(r'I', fontsize=34, rotation=0, position=(0, 0.32))
        plt.xlabel(r'sub-channel index', fontsize=34)
        plt.xticks([0, 58, 117, 175, 233], fontsize=26)  # np.arange(0, num_subcarriers+1, 18))
        locs, labels = plt.yticks()
        plt.yticks([-1, 0, 1], fontsize=26)
        plt.tight_layout()
        name_fig = './plots/plot_re_' + name_file[:-4] + '_tx' + str(tx_ant) + '.pdf'
        plt.savefig(name_fig)
        plt.close()

        fig = plt.figure(constrained_layout=False)
        fig.set_size_inches(6, 3)
        plt.plot(np.imag(v_matrix[sample_idx, rx_ant, :, tx_ant]), 'purple', linewidth=2.5)
        plt.grid()
        plt.ylabel(r'Q', fontsize=34, rotation=0, position=(0, 0.32))
        plt.xlabel(r'sub-channel index', fontsize=34)
        plt.xticks([0, 58, 117, 175, 233], fontsize=26)  # np.arange(0, num_subcarriers+1, 18))
        locs, labels = plt.yticks()
        plt.yticks([-1, 0, 1], fontsize=26)
        plt.tight_layout()
        name_fig = './plots/plot_im_' + name_file[:-4] + '_tx' + str(tx_ant) + '.pdf'
        plt.savefig(name_fig)
        plt.close()
