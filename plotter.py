# -*- coding: utf-8 -*-
"""
|/--------------------TITLE--------------------\|
PHYS20161 -- Assignment N -- [ASSIGNMENT NAME]
|/---------------------------------------------\|

[CODE DESCRIPTION/USAGE/OUTLINE]


Created: Fri Mar 10 15:08:27 2023
Last Updated:

@author: Charlie Fynn Perkins, UID: 10839865 0
"""
import numpy as np
import matplotlib.pyplot as plt
import calculator as calc

# File Settings
FILENAME = "energy_function_combined_append"
DIRECTORY = ".\\saves\\"
EXTENSION = ".txt"
DELIMITER = ","
SKIP_HEADER = 1
SKIP_FOOTER = 0

# Data Settings
IND_VAR = 1
IND_UNITS = ["Pressure, Pa"]
DEP_VAR = 1
DEP_UNITS = ["Energy Density, Pa"]


def get_path(directory, filename, extension):
    directory, filename, extension = str(
        directory), str(filename), str(extension)
    return directory + filename + extension


def get_data(path, delimiter, skip_head, skip_foot):
    data = np.genfromtxt(path, dtype=float, delimiter=",", comments='#',
                         skip_header=skip_head, skip_footer=skip_foot)
    return data


def plot_data(data, ideal_filename, ind_var, ind_units, dep_var, dep_units):
    # init_ind_var = 0
    # init_dep_var = 0
    # for col in range(ind_var):
    #     ind_var_plot = data[:, col]
    ind_var_plot = data[:, 0]
    dep_var_plot = data[:, 1]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # if len(dep_units > 1):
    #     ax2 = ax.twinx()
    ax.set(xlabel=ind_units[0])
    ax.set(ylabel=dep_units[0])
    ax.plot(ind_var_plot, dep_var_plot, color="red",
            marker=".", linestyle='None')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc='upper left')
    ax.grid()
    plt.tight_layout()
    filename = calc.path_checker(ideal_filename, ".png")
    print("Saving figure...")
    plt.savefig(filename, dpi=1000)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    path = get_path(DIRECTORY, FILENAME, EXTENSION)
    data = get_data(path, DELIMITER, SKIP_HEADER, SKIP_FOOTER)
    plot_data(data, FILENAME, IND_VAR, IND_UNITS, DEP_VAR, DEP_UNITS)
