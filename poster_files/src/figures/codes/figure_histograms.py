import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import PyThemes.quarto as quarto
import svgutils.transform as sg

path_gen = "/media/s88422cg/Backup_dune/DUNE/PhD_Parts/Part5_Winds/Giant_dune_retroaction_regional_wind_regime"
sys.path.append(path_gen)
from python_codes.plot_functions import make_nice_histogram

# plt.rcParams["figure.constrained_layout.h_pad"] = 0.005
plt.rcParams["figure.constrained_layout.w_pad"] = 0

plt.rcParams["text.latex.preamble"] = r"\usepackage{fontawesome5}"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = "medium"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.top"] = True

color_day = "#C3B632"
color_night = "#3D134F"
color_Era5Land = "tab:orange"
color_insitu = "tab:blue"
color_dune_orientation = "grey"

# paths
path_data = os.path.join(path_gen, "static/data/processed_data")


# Loading wind data
Data = np.load(os.path.join(path_data, "Data_final.npy"), allow_pickle=True).item()
Stations = sorted(Data.keys())

# Figure properties
Stations = ["Deep_Sea_Station", "South_Namib_Station"]
#
theta_bins_list = [[[0, 90], [150, 230]], [[0, 140], [150, 260]]]
velocity_bins_list = [[[0.05, 0.2], [0.3, 10]], [[0.05, 0.2], [0.3, 10]]]
Data_pattern = np.load(
    os.path.join(path_data, "Data_DEM.npy"), allow_pickle=True
).item()
icon = [r"\faSun", r"\faMoon"]
labels = [r"\textbf{a}", r"\textbf{b}"]

color_ax = "purple"

ind = 0
station = Stations[ind]
theta_bins = theta_bins_list[ind]
velocity_bins = velocity_bins_list[ind]

# ################ Figure
fig, axarr = plt.subplots(
    2,
    2,
    layout="constrained",
    figsize=np.array(
        [0.45 * quarto.regular_fig_width, 0.225 * quarto.regular_fig_width]
    )
    * 1.6,
)

for i in range(2):  # Loop over velocites
    mask_U = (Data[station]["U_star_era"] >= velocity_bins[i][0]) & (
        Data[station]["U_star_era"] <= velocity_bins[i][1]
    )
    label_u = (
        r"$u_{*, \, \text{ERA}} < " + str(velocity_bins[i][1]) + "$ \n m s$^{-1}$"
        if i == 0
        else r"$u_{*, \, \text{ERA}} > " + str(velocity_bins[i][0]) + "$ \n m s$^{-1}$"
    )

    axarr[i, -1].set_ylabel(label_u)
    axarr[i, -1].yaxis.set_label_position("right")
    for j in range(2):  # loop over angles
        mask_theta = (Data[station]["Orientation_era"] >= theta_bins[j][0]) & (
            Data[station]["Orientation_era"] <= theta_bins[j][1]
        )
        label_theta = (
            icon[j]
            + "\n"
            + rf"${theta_bins[j][0]:d}^\circ <"
            + r"\theta_{ERA}"
            + rf"< {theta_bins[j][-1]:d}^\circ $"
        )

        make_nice_histogram(
            Data[station]["Orientation_insitu"][mask_theta & mask_U],
            80,
            axarr[i, j],
            alpha=0.5,
            color=color_insitu,
        )
        make_nice_histogram(
            Data[station]["Orientation_era"][mask_theta & mask_U],
            80,
            axarr[i, j],
            alpha=0.5,
            color=color_Era5Land,
        )
        #
        axarr[i, j].axvline(
            Data_pattern[station]["orientation"],
            color=color_dune_orientation,
            ls="--",
            lw=2,
        )
        axarr[i, j].axvline(
            (Data_pattern[station]["orientation"] + 180) % 360,
            color=color_dune_orientation,
            ls="--",
            lw=2,
        )
        #
        if i == 0:
            axarr[i, j].set_xlabel(label_theta, usetex=True)
            axarr[i, j].xaxis.set_label_position("top")
            if j == 1:
                for axis in ["top", "bottom", "left", "right"]:
                    axarr[i, j].spines[axis].set_color(color_ax)
                    axarr[i, j].spines[axis].set_linewidth(2)

    for ax in axarr.flatten():
        ax.set_yticks([])
    for ax in axarr[0, :]:
        ax.set_xticklabels([])
    for ax in axarr[-1, :]:
        ax.set_xticks([45, 125, 215, 305])
    fig.supxlabel(r"Wind direction, $\theta~[^\circ]$")
    fig.supylabel("Counts")

fig.patch.set_alpha(0)

# %% saving figure
figname = "../{}.svg".format(sys.argv[0].split(os.sep)[-1].replace(".py", ""))
fig.savefig(figname, dpi=1200)

fig = sg.fromfile(figname)
newsize = [
    "{}pt".format(quarto.scaling_factor * float(i.replace("pt", "")))
    for i in fig.get_size()
]

fig.set_size(newsize)
fig.save(figname)
