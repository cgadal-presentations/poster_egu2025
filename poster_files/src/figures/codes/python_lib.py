import matplotlib.colors as mpcolors
import matplotlib.patches as ptch
import numpy as np


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncate a given colormap.

    Parameters
    ----------
    cmap : matplotlib colormap.
        Input colormap to be truncated.
    minval : float
        Where to start the wanted colormap, assuming that 0 is the start of the input colormap, and 1 its end (the default is 0.0).
    maxval : type
        Where to start the wanted colormap, assuming that 0 is the start of the input colormap, and 1 its end (the default is 0.0).
    n : type
        Description of parameter `n` (the default is 100).

    Returns
    -------
    type
        Description of returned object.

    """
    # from here: https://stackoverflow.com/a/18926541/9530017
    new_cmap = mpcolors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def north_arrow(
    ax,
    center,
    length,
    length_small=None,
    width=None,
    radius=None,
    theta=0,
    textcolor="k",
    transform=None,
    **kwargs,
):
    """Plot a arrow indicating the North on a figure.

    Parameters
    ----------
    ax : matplotlib axe
        Axe on which to plot the arrow
    center : list, tuple, np.array
        Position of the arrow
    length : float
        arrow max length
    length_small : float
        length of the center par tof the arrow (the default is 0.8*length).
    width : float
        arrow width (the default is (3/7)*length).
    radius : float
        distance between the text and the arrow (the default is (45/70)*length).
    theta : float
        rotation of the arrow indicating the north (the default is 0 for an arrow pointing upward).
    textcolor : str
        color of the text (the default is 'k').
    transform : matplotlib transform
        transform for the coordinate systen of the input length and positions (the default is ax.transData).
    **kwargs :
        Optional parameters passed to :class:`Polygon <matplotlib.patches.Polygon>`, used to customize the arrow.

    Returns
    -------
    None
        return nothing

    """
    if transform is None:
        transform = ax.transData
    if length_small is None:
        length_small = 0.8 * length
    if width is None:
        width = (3 / 7) * length
    if radius is None:
        radius = (45 / 70) * length
    y_start = radius + length - length_small
    arrow = np.array(
        [
            [0, y_start],
            [width / 2, radius],
            [0, radius + length],
            [-width / 2, radius],
            [0, y_start],
        ]
    )
    # barycentre = np.sum(arrow, axis=0)/arrow.shape[0]
    # arrow = np.dot(Rotation_matrix(theta), (arrow-barycentre).T).T + barycentre
    r = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    arrow = np.dot(r, arrow.T).T
    arrow = arrow + np.array(center)
    #
    ax.add_patch(ptch.Polygon(arrow, transform=transform, **kwargs))
    ax.text(
        center[0],
        center[1],
        r"N",
        transform=transform,
        ha="center",
        va="center",
        fontweight="bold",
    )
