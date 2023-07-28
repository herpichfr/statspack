#!/usr/bin/env python3
# package with modules for statistic visualization
# Fabio R. Herpich 2023-07-26 CASU/IoA Cambridge
# All rights reserved (see LICENSE file)

import numpy as np
import logging
import colorlog
from datetime import datetime
import matplotlib.pyplot as plt
from .find_confidence_interval import find_confidence_interval


def call_logger(name, level=logging.INFO):
    """Configure the logger."""
    logging.shutdown()
    logging.root.handlers.clear()

    logger = colorlog.getLogger(name)
    logger.setLevel(level)

    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s:%(name)s:%(message)s',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'blue',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        })

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def density_contour(xdata, ydata, binsx, binsy, ax=None, fill=False, levels_prc=[.68, .95, .99], verbose=False, **contour_kwargs):
    """ Create a density contour plot.

    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    binsx : int
        Number of bins along x dimension
    binsy : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    fill : bool
        If True, fill the contours
    levels_prc : list
        List of confidence levels to plot. Default is [.68, .95, .99]
    verbose : bool
        verbose
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()

    Returns
    -------
    contours, levels : tuple
        A tuple containing the contour plot(s) and the calculated contour levels.
    """
    __name__ = 'statspack.density_contour'
    call_logger(__name__, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Initializing density_contour"]))

    H, xedges, yedges = np.histogram2d(
        xdata, ydata, bins=[binsx, binsy], density=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1])
    y_bin_sizes = (yedges[1:] - yedges[:-1])

    if verbose:
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Calculating contour levels"]))
    pdf = (H * (x_bin_sizes * y_bin_sizes))

    if verbose:
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Calculating confidence intervals"]))
    levels = [find_confidence_interval(
        pdf, prc, verbose=verbose) for prc in levels_prc]

    X, Y = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
    Z = pdf.T

    if verbose:
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Creating contours"]))
    if ax is None:
        ax = plt.gca()

    contour = ax.contour(X, Y, Z, levels=levels,
                         origin="lower", **contour_kwargs)
    out = (contour,)

    if fill:
        contourf = ax.contourf(X, Y, Z, levels=levels,
                               origin="lower", **contour_kwargs)
        out = (contour, contourf)

    return out, levels
