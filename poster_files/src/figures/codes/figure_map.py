"""

============
Figure 1
============

"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import PyThemes.quarto as quarto
import svgutils.transform as sg
from PIL import Image
from python_lib import north_arrow

global_path = "/media/s88422cg/Backup_dune/DUNE/PhD_Parts/Part5_Winds/Giant_dune_retroaction_regional_wind_regime/"

# paths
path_images = os.path.join(global_path, "static/images/")
path_savefig = os.path.join(global_path, "Paper/Figures")
path_data = os.path.join(global_path, "static/data/processed_data")

# Loading wind data
Data = np.load(os.path.join(path_data, "Data_final.npy"), allow_pickle=True).item()
Stations = sorted(Data.keys())

# images
list_images = sorted(glob.glob(os.path.join(path_images, "*+*")))
order_plot = [1, 2, 4, 6, 6]

# fig properties
bins = [0.03, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
labels = ["Etosha West", "Huab", "North Sand Sea", "South Sand Sea"]
coords_stations = np.array(
    [
        (-19.034111, 15.737194),
        (-20.874722, 13.642),
        (-24.125533, 15.049100),
        (-26.044083, 15.396972),
    ]
)
scales = [1300, 1100, 1650, 2600]
bbox = dict(facecolor=(1, 1, 1, 0.5), edgecolor=(1, 1, 1, 0))
bbox2 = dict(facecolor=(1, 1, 1, 0.5), edgecolor=(1, 1, 1, 0), pad=0.75)
numbering = [
    r"$\quad$\textbf{a}",
    r"\textbf{b}",
    r"\textbf{c}",
    r"\textbf{d}",
    r"\textbf{e}",
]
coords_insitu_pix = [(1141, 544), (881, 554), (755, 430), (772, 550)]

# #### Figure
fig, ax0 = plt.subplots(
    1,
    1,
    layout="constrained",
    figsize=(0.45 * quarto.regular_fig_width, 0.75 * quarto.regular_fig_width),
)


#
# # map
Map = np.array(Image.open(os.path.join(path_images, "Map.png")))
ax0.imshow(Map[:-104, 642:-791], extent=[12.55, 17.38, -27.27, -18.2])
ax0.set_xlabel(r"Longitude [$^\circ$]")
ax0.set_ylabel(r"Latitude [$^\circ$]")
ax0.yaxis.set_label_position("right")
ax0.yaxis.tick_right()
#
ax0.scatter(
    coords_stations[:, 1],
    coords_stations[:, 0],
    s=25,
    color="k",
)
for point, txt in zip(coords_stations, labels):
    pad_x, pad_y = 0, 0
    if "Sand Sea" in txt:
        ha, va = "center", "top"
        pad_y = -0.15
    elif txt == "Huab":
        ha, va = "left", "center"
        pad_x = 0.15
    else:
        ha, va = "right", "center"
        pad_x = -0.15
    ax0.annotate(
        txt,
        (point[1] + pad_x, point[0] + pad_y),
        ha=ha,
        va=va,
        color="k",
        bbox=bbox2,
        fontweight="bold",
    )

# north arrow
rect = plt.Rectangle(
    (0.90, 0.86), width=0.1, height=0.3, color="w", alpha=0.4, transform=ax0.transAxes
)
ax0.add_patch(rect)
center = np.array([0.95, 0.9])
length = 0.05
north_arrow(
    ax0, center, length, width=0.9 * length, transform=ax0.transAxes, color="k", lw=0.05
)


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
