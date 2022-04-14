
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times'
rcParams['text.usetex'] = 'true'
rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
rcParams['font.size'] = 16


def plt_confusion_matrix(number_activities, confusion_matrix, activities, name):
    confusion_matrix_normaliz_row = np.transpose(confusion_matrix / np.sum(confusion_matrix, axis=1).reshape(-1, 1))
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(5, 4)
    ax = fig.add_axes((0.18, 0.15, 0.6, 0.8))
    im1 = ax.pcolor(np.linspace(0.5, number_activities + 0.5, number_activities + 1),
                    np.linspace(0.5, number_activities + 0.5, number_activities + 1),
                    confusion_matrix_normaliz_row, cmap='Purples', edgecolors='black', vmin=0, vmax=1)
    ax.set_xlabel('actual ID', FontSize=14)
    ax.set_xticks(np.linspace(1, number_activities, number_activities))
    ax.set_xticklabels(labels=activities, FontSize=12)
    ax.set_yticks(np.linspace(1, number_activities, number_activities))
    ax.set_yticklabels(labels=activities, FontSize=12, rotation=45)
    ax.set_ylabel('predicted ID', FontSize=14)

    for x_ax in range(confusion_matrix_normaliz_row.shape[0]):
        for y_ax in range(confusion_matrix_normaliz_row.shape[1]):
            col = 'k'
            value_c = round(confusion_matrix_normaliz_row[x_ax, y_ax], 2)
            if value_c > 0.6:
                col = 'w'
            ax.text(y_ax + 1, x_ax + 1, '%.2f' % value_c, horizontalalignment='center',
                    verticalalignment='center', fontsize=10, color=col)

    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.8])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.ax.set_ylabel('Accuracy', FontSize=14)
    cbar.ax.tick_params(axis="y", labelsize=11)

    plt.tight_layout()
    name_fig = './plots/cm_' + name + '.pdf'
    plt.savefig(name_fig)
