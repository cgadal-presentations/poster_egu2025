import os
import sys

import cmasher as cmr
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import numpy as np
import PyThemes.quarto as quarto
import svgutils.transform as sg

path_gen = "/media/s88422cg/Backup_dune/DUNE/PhD_Parts/Part5_Winds/Giant_dune_retroaction_regional_wind_regime"
sys.path.append(path_gen)

from python_codes.general import find_mode_distribution, smallestSignedAngleBetween
from python_codes.plot_functions import plot_regime_diagram

plt.rcParams["figure.constrained_layout.hspace"] = 0
plt.rcParams["figure.constrained_layout.h_pad"] = 0
plt.rcParams["figure.constrained_layout.w_pad"] = 0

plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = "medium"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.top"] = True

# paths
path_data = os.path.join(path_gen, "static/data/processed_data")

# ##### Loading meteo data
Data = np.load(os.path.join(path_data, "Data_final.npy"), allow_pickle=True).item()
Stations = ["South_Namib_Station", "Deep_Sea_Station"]

# #### Computing quantities

Orientation_era = np.concatenate(
    [Data[station]["Orientation_era"] for station in Stations]
)
Orientation_insitu = np.concatenate(
    [Data[station]["Orientation_insitu"] for station in Stations]
)
U_era = np.concatenate([Data[station]["U_star_era"] for station in Stations])
U_insitu = np.concatenate([Data[station]["U_star_insitu"] for station in Stations])
numbers = {
    key: np.concatenate([Data[station][key] for station in Stations])
    for key in ("Froude", "kH", "kLB")
}
#
Delta = smallestSignedAngleBetween(Orientation_era, Orientation_insitu)
mode_delta = np.array(
    [find_mode_distribution(Delta, i) for i in np.arange(150, 350)]
).mean()
delta_angle = np.abs(Delta)
delta_u = (U_era - U_insitu) / U_era

# #### Figure parameters
cmap_delta_u = cmr.prinsenvlag_r
colors = cmap_delta_u(np.linspace(0.5, 1, cmap_delta_u.N))
cmap_delta_theta = mpcolors.LinearSegmentedColormap.from_list(
    "prinsenvlag_r_pos", colors
)
lims = {"Froude": (5.8e-3, 450), "kLB": (0.009, 7.5), "kH": (2.2e-2, 10.8)}
cmaps = [cmap_delta_theta, cmap_delta_u]
regime_line_color = "tab:green"
norms = [
    mpcolors.Normalize(vmin=0, vmax=99),
    mpcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=1),
]
cbar_labels = [r"Deflection $[^\circ]$", r"Rel. velocity difference"]

quantities = [delta_angle, delta_u]
labels = [r"\textbf{a}", r"\textbf{b}"]
cbticks = [[0, 25, 50, 75], [-3, -1.5, 0, 0.5, 1]]

mask = ~np.isnan(numbers["Froude"])

var1, var2 = "Froude", "kH"
xlabel = r"$\mathcal{F} =  U/\sqrt{(\delta\rho/\rho_{0}) g H}$"

# #### Figure
fig, axarr = plt.subplots(
    1,
    2,
    figsize=np.array([0.45 * quarto.regular_fig_width, 0.3 * quarto.regular_fig_width])
    * 1.2,
    layout="constrained",
)

for i, (ax, quantity, cmap, norm, cbtick) in enumerate(
    zip(axarr.flatten(), quantities, cmaps, norms, cbticks)
):
    ylabel = "$k H$" if i == 0 else None
    #
    vars = [numbers[var1][mask], numbers[var2][mask]]
    lims_list = [lims[var1], lims[var2]]
    #
    bin1 = np.logspace(
        np.floor(np.log10(numbers[var1][mask].min())),
        np.ceil(np.log10(numbers[var1][mask].max())),
        50,
    )
    bin2 = np.logspace(
        np.floor(np.log10(numbers[var2][mask].min())),
        np.ceil(np.log10(numbers[var2][mask].max())),
        50,
    )
    bins = [bin1, bin2]
    a = plot_regime_diagram(
        ax,
        quantity[mask],
        vars,
        lims_list,
        xlabel,
        ylabel,
        bins=bins,
        norm=norm,
        cmap=cmap,
        type="binned",
    )
    #
    # regime lines
    ax.axvline(0.4, color=regime_line_color, linestyle="--", lw=2)
    ax.axhline(0.32, color=regime_line_color, linestyle="--", lw=2)
    #
    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, ax=ax, location="top", ticks=cbtick, pad=0)
    cb.set_label(cbar_labels[i])
    #
    ax.set_xticks([0.1, 10])
    ax.set_ylabel(ylabel, labelpad=0)
    ax.set_xlabel(xlabel, labelpad=0)

fig.align_labels()

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
