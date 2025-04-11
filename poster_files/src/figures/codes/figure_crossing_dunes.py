import glob
import os
import sys

import matplotlib.colors as mpcolors
import matplotlib.patches as ptch
import matplotlib.path as path
import matplotlib.pyplot as plt
import numpy as np
import PyThemes.quarto as quarto
import svgutils.transform as sg
from matplotlib.colors import to_rgba

path_gen = "/media/s88422cg/Backup_dune/DUNE/PhD_Parts/Part5_Winds/Giant_dune_retroaction_regional_wind_regime"
sys.path.append(path_gen)

from python_codes.CourrechDuPont2014 import (
    Bed_Instability_Orientation,
    Elongation_direction,
)
from python_codes.general import Make_angular_PDF, Vector_average, cosd, sind
from python_codes.meteo_analysis import quadratic_transport_law, quartic_transport_law
from python_codes.plot_functions import plot_flux_rose


def North_arrow(
    fig, ax, center, length, length_small, width, radius, theta=0, color="k"
):
    y_start = radius + length - length_small
    arrow = np.array(
        [
            [0, -y_start],
            [width / 2, -radius],
            [0, -(radius + length)],
            [-width / 2, -radius],
            [0, -y_start],
        ]
    )
    arrow = arrow + np.array(center)
    ax.add_patch(plt.Polygon(arrow, color=color))
    t = ax.text(center[0], center[1], "N")
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bb = t.get_window_extent(renderer=r).transformed(inv)
    t.set_visible(False)
    height_t = bb.height
    t = ax.text(
        center[0],
        center[1] - height_t / 2,
        r"N",
        color=color,
        ha="center",
        fontweight="bold",
    )


# color_F = "#AF1B3F"
# color_BI = "#157A6E"

color_F = "#9b2226"
color_BI = "#0a9396"
#
linewidth_barscale = 2
lw_arrow = 1.5
flux_color = mpcolors.ListedColormap("navajowhite")


def plot_arrow(ax, point, length, angle, type, arrowprops):
    dx = cosd(angle) * length
    dy = sind(-angle) * length
    # dx = int(round(cosd(angle)*length))
    # dy = int(round(sind(-angle)*length))
    if type == "centered":
        xy = point - np.array([dx, dy]) / 2
        xytext = point + np.array([dx, dy]) / 2
    else:
        xy = point
        xytext = xy + np.array([dx, dy])
    arrow = ptch.FancyArrowPatch(xytext, xy, **arrowprops)
    ax.add_patch(arrow)
    if arrow.get_linestyle() != "-":
        # Tail
        v1 = arrow.get_path().vertices[0:3, :]
        c1 = arrow.get_path().codes[0:3]
        p1 = path.Path(v1, c1)
        pp1 = ptch.PathPatch(
            p1,
            color=arrow.get_facecolor(),
            lw=arrow.get_linewidth(),
            linestyle=arrow.get_linestyle(),
            fill=False,
        )
        ax.add_patch(pp1)
        # Heads ====> partie qui ne marche pas
        v2 = arrow.get_path().vertices[3:, :]
        c2 = arrow.get_path().codes[3:]
        c2[0] = 1
        p2 = path.Path(v2, c2)
        pp2 = ptch.PathPatch(
            p2, color=arrow.get_facecolor(), lw=arrow.get_linewidth(), linestyle="-"
        )
        ax.add_patch(pp2)
        arrow.remove()


def plot_orientation_wedge(
    ax, A_F, A_BI, center, length, color_F, color_BI, alpha=0.2, **kwargs
):
    wedge_f = ptch.Wedge(
        center,
        length,
        -np.nanmax(A_F),
        -np.nanmin(A_F),
        edgecolor=color_F,
        facecolor=to_rgba(color_F, alpha),
        joinstyle="round",
        **kwargs,
    )
    wedge_bi = ptch.Wedge(
        center,
        length,
        -np.nanmax(A_BI),
        -np.nanmin(A_BI),
        edgecolor=color_BI,
        facecolor=to_rgba(color_BI, alpha),
        joinstyle="round",
        **kwargs,
    )
    wedge_bis = ptch.Wedge(
        center,
        length,
        -np.nanmax(A_BI) + 180,
        -np.nanmin(A_BI) + 180,
        edgecolor=color_BI,
        facecolor=to_rgba(color_BI, alpha),
        joinstyle="round",
        **kwargs,
    )
    #
    ax.add_patch(wedge_f)
    ax.add_patch(wedge_bi)
    ax.add_patch(wedge_bis)


# path
path_imgs = path_gen + "/static/images/"
path_outputdata = path_gen + "/static/data/processed_data/"

# ##### Loading meteo data
Data = np.load(
    os.path.join(path_outputdata, "Data_final.npy"), allow_pickle=True
).item()
station = "Deep_Sea_Station"

# ##### Calculation of sediment flux rose and dune orientations
rho_g = 2.65e3  # grain density
rho_f = 1  # fluid density
g = 9.81  # [m/s2]
grain_diameters = np.linspace(100e-6, 400e-6, 30)  # grain size [m]
Q = (
    np.sqrt((rho_g - rho_f * g * grain_diameters) / rho_f) * grain_diameters
)  # characteristic flux [m2/s]
#
# Quadratic transport law parameters
theta_th_quadratic = 0.005  # threshold shield numbers for the quadratic
Omega = 8
# Quartic transport law parameters
theta_th_quartic = 0.0035  # threshold shield numbers for the quartic
#
gamma = np.array(list(np.logspace(-1, 1, 10)) + [1.6])

# Vector of orientations and shear velocity
Orientations = np.array(
    [Data[station]["Orientation_insitu"], Data[station]["Orientation_era"]]
)
Shear_vel = np.array([Data[station]["U_star_insitu"], Data[station]["U_star_era"]])
# corresponding shield number
theta = (rho_f / ((rho_g - rho_f) * g * grain_diameters[:, None, None])) * Shear_vel[
    None, :, :
] ** 2
# sediment fluxes
q = np.array(
    [
        quadratic_transport_law(theta, theta_th_quadratic, Omega),
        quartic_transport_law(theta, theta_th_quartic),
    ]
)
# Angular distributions of sediment fluxes
PDF, Angles = Make_angular_PDF(Orientations[None, None, :, :] * np.ones(q.shape), q)
# Dune orientations
alpha_BI = Bed_Instability_Orientation(
    Angles[None, None, None, None, :],
    PDF[None, :, :, :, :],
    gamma=gamma[:, None, None, None, None],
)
alpha_F = Elongation_direction(
    Angles[None, None, None, None, :],
    PDF[None, :, :, :, :],
    gamma=gamma[:, None, None, None, None],
)


# ### figure properties
index = 8
fontsize_small = 8

start_v = 200
height = 800
left_point = 0
aspect_ratio = 2.3
path_im = "/media/s88422cg/Backup_dune/DUNE/PhD_Parts/Thèse/Chapitres/chapter5/figures/Figure_small_scale_finger/"
img = np.array(plt.imread(glob.glob(path_im + "*.png")[0]))[
    start_v : start_v + height, left_point : left_point + int(aspect_ratio * height), :
]

# Figure
figsize = (1.48 * quarto.regular_fig_width, quarto.regular_fig_width / aspect_ratio)
fig, ax = plt.subplots(1, 1, layout="compressed", figsize=figsize)

# image
ax.imshow(img, alpha=0.9)
# plt.axis('off')
ax.set_xticks([])
ax.set_yticks([])

# échelle
backgrnd = ptch.Rectangle(
    (0, 0), width=0.16, height=0.11, transform=ax.transAxes, color="w", alpha=0.6
)
ax.add_patch(backgrnd)
plt.plot([30, 30 + 230], [780, 780], linewidth=linewidth_barscale, color="k")
name = r"630 m"
plt.text(30 + 230 / 2, 765, name, color="k", ha="center")

# #### custom legend
legend_elements = [
    plt.Line2D([0], [0], color=color_BI, label="Bed instability"),
    plt.Line2D([0], [0], color=color_F, label="Elongation mechanism"),
    plt.Line2D([0], [0], color="k", label="regional winds"),
    plt.Line2D([0], [0], color="k", ls="--", label="local winds"),
]

leg = ax.legend(
    handles=legend_elements,
    bbox_to_anchor=(-0.01, 0),
    loc="lower right",
    title="Predicted dune orientations",
    borderaxespad=0.0,
    ncol=2,
    handlelength=1,
    columnspacing=1.25,
    handletextpad=0.6,
)
title = leg.get_title()
title.set_weight("bold")
leg.get_frame().set_facecolor("none")
leg.get_frame().set_edgecolor("k")


# ######### dune types
# bed instability
ellispe3 = ptch.Ellipse((1025, 380), 250, 790, angle=0, color=color_F, fill=False)
ax.add_artist(ellispe3)
# ellispe2 = ptch.Ellipse((1110, 453), 150, 100, angle = 0, color = color_BI, alpha = 0.4, fill = False)
# ax.add_artist(ellispe2)
# big finger
ellispe1 = ptch.Ellipse((71, 550), 170, 250, angle=0, color=color_BI, fill=False)
ax.add_artist(ellispe1)

# small fingers
ellispe4 = ptch.Ellipse(
    (277, 533), 400, 40, angle=-35, color=color_F, fill=False, linestyle="--"
)
ax.add_artist(ellispe4)
ellispe5 = ptch.Ellipse(
    (1323, 500), 350, 50, angle=-35, color=color_F, fill=False, linestyle="--"
)
ax.add_artist(ellispe5)

# flux rose era
anchor = [0.15, 0.64, 0.45, 0.45]
subax = ax.inset_axes(bounds=anchor, transform=ax.transAxes)
a = plot_flux_rose(
    Angles,
    PDF[0, index, 1, :],
    subax,
    fig,
    cmap=flux_color,
    edgecolor="k",
    linewidth=0.5,
)
#
RDD, RDP = Vector_average(Angles, PDF[0, index, 1, :])
a.annotate(
    "",
    (RDD * np.pi / 180, 0),
    (RDD * np.pi / 180, 0.85 * a.get_rmax()),
    arrowprops=dict(arrowstyle="<|-", shrinkA=0, shrinkB=0, color="saddlebrown"),
)
a.grid(linewidth=0.4, color="k", linestyle="--")
a.set_axisbelow(True)
a.patch.set_alpha(0.4)
a.set_xticklabels([])
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
a.text(
    0.5,
    0.2,
    "Sand flux \u2014 regional",
    transform=a.transAxes,
    ha="center",
    verticalalignment="top",
    bbox=props,
    fontsize=fontsize_small,
    # multialignment="center",
)

length = 400 / 2
center = (650, 324)

plot_arrow(
    ax,
    center,
    2 * length,
    alpha_BI[-1, 0, index, 1],
    "centered",
    arrowprops=dict(
        arrowstyle="<|-|>",
        color=color_BI,
        shrinkA=0,
        shrinkB=0,
        lw=lw_arrow,
        mutation_scale=10,
        linestyle="-",
    ),
)
plot_arrow(
    ax,
    center,
    length,
    alpha_F[-1, 0, index, 1],
    "not_centered",
    arrowprops=dict(
        arrowstyle="<|-",
        color=color_F,
        shrinkA=0,
        shrinkB=0,
        lw=lw_arrow,
        mutation_scale=10,
        linestyle="-",
    ),
)
plot_orientation_wedge(
    ax,
    alpha_F[:, :, :, 1],
    alpha_BI[:, :, :, 1],
    center,
    length,
    color_F,
    color_BI,
    alpha=0.2,
)

# flux rose station
anchor = [0.8, 0.64, 0.45, 0.45]
subax = ax.inset_axes(bounds=anchor, transform=ax.transAxes)
a = plot_flux_rose(
    Angles,
    PDF[0, index, 0, :],
    subax,
    fig,
    cmap=flux_color,
    edgecolor="k",
    linewidth=0.5,
)
#
RDD, RDP = Vector_average(Angles, PDF[0, index, 0, :])
a.annotate(
    "",
    (RDD * np.pi / 180, 0),
    (RDD * np.pi / 180, 0.85 * a.get_rmax()),
    arrowprops=dict(arrowstyle="<|-", shrinkA=0, shrinkB=0, color="saddlebrown"),
)
a.grid(linewidth=0.4, color="k", linestyle="--")
a.set_axisbelow(True)
a.patch.set_alpha(0.4)
a.set_xticklabels([])
#
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
a.text(
    0.5,
    0.2,
    "Sand flux \u2014 local",
    transform=a.transAxes,
    ha="center",
    va="top",
    bbox=props,
    fontsize=fontsize_small,
    multialignment="center",
)

a.spines["polar"].set_linestyle("--")


center = (1480, 670)

plot_arrow(
    ax,
    center,
    2 * length,
    alpha_BI[-1, 0, index, 0],
    "centered",
    arrowprops=dict(
        arrowstyle="<|-|>",
        color=color_BI,
        shrinkA=0,
        shrinkB=0,
        lw=lw_arrow,
        mutation_scale=10,
        linestyle="--",
    ),
)
plot_arrow(
    ax,
    center,
    length,
    alpha_F[-1, 0, index, 0],
    "not_centered",
    arrowprops=dict(
        arrowstyle="<|-",
        color=color_F,
        shrinkA=0,
        shrinkB=0,
        lw=lw_arrow,
        mutation_scale=10,
        linestyle="--",
    ),
)
plot_orientation_wedge(
    ax,
    alpha_F[:, :, :, 0],
    alpha_BI[:, :, :, 0],
    center,
    length,
    color_F,
    color_BI,
    alpha=0.2,
    linestyle="--",
)

length = 70
length_small = 0.8 * length
width = 40
radius = 35
center = np.array([1800, 755])
#
backgrnd = ptch.Rectangle(
    (0.96, 0), width=0.04, height=0.2, transform=ax.transAxes, color="w", alpha=0.6
)
ax.add_patch(backgrnd)
North_arrow(fig, ax, center, length, length_small, width, radius, theta=0, color="k")
#

ax.patch.set_alpha(0)
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
