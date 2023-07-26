#!/bin/env python3
# package with modules for statistic visualization
# herpich 2022-12-20 fabiorafaelh@gmail.com

import numpy as np
import logging
import colorlog
from datetime import datetime
from scipy.stats import kde, scoreatpercentile
import scipy.stats
# from scipy.stats import gaussian_kde, scoreatpercentile
from multiprocessing import Pool
import matplotlib.pyplot as plt


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


def bining(x, y, z, nbins=10, xlim=(None, None), ylim=(None, None)):
    """Bining data for contour plots.

    Parameters
    ----------
    x : numpy.ndarray
        x data
    y : numpy.ndarray
        y data
    z : numpy.ndarray
        z data
    nbins : int
        number of bins
    xlim : tuple
        x limits
    ylim : tuple
        y limits

    Returns
    -------
    X : numpy.ndarray
        x data binned
    Y : numpy.ndarray
        y data binned
    Z : numpy.ndarray
        z data binned
    """
    # initialize the logger
    call_logger(__name__, level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Check for valid xlim and ylim inputs
    if xlim[0] is None:
        xlim = (x.min(), x.max())
    if ylim[0] is None:
        ylim = (y.min(), y.max())

    # Compute bin edges
    xv = np.linspace(xlim[0], xlim[1], nbins + 1)
    yv = np.linspace(ylim[0], ylim[1], nbins + 1)

    # Find bin indices for each data point using digitize
    x_bins = np.digitize(x, xv) - 1
    y_bins = np.digitize(y, yv) - 1

    # Calculate the bin centers using the digitized indices
    logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Calculating bin centers"]))
    x_center = (xv[:-1] + xv[1:]) / 2
    y_center = (yv[:-1] + yv[1:]) / 2

    # Create a 2D array of bin indices for each data point
    bin_indices = x_bins * nbins + y_bins

    # Initialize empty arrays for the binned data
    X = np.full(nbins**2, np.nan)
    Y = np.full(nbins**2, np.nan)
    Z = np.full(nbins**2, np.nan)

    # Calculate medians for each bin using bin_indices
    logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Calculating medians for the bins"]))
    for i in range(nbins**2):
        mask = (bin_indices == i)
        bin_data = z[mask]
        if len(bin_data) > 0:
            Z[i] = np.median(bin_data)

    # Unravel the 1D arrays to 2D arrays for X, Y, and Z
    X = x_center.repeat(nbins)
    Y = np.tile(y_center, nbins)

    return X, Y, Z


# def find_confidence_interval(x, pdf, confidence_level):
#     """Find confidence interval from PDF.
#
#     Parameters
#     ----------
#     x : numpy.ndarray
#         x data
#     pdf : numpy.ndarray
#         pdf data
#     confidence_level : float
#         confidence level
#
#     Returns
#     -------
#     out : float
#         confidence interval
#     """
#     return pdf[pdf > x].sum() - confidence_level
#
#
# def density_contour(xdata, ydata, binsx, binsy, ax=None, range=None, fill=False, levels_prc=[.68, .95, .99], **contour_kwargs):
#     """ Create a density contour plot.
#
#    Parameters
#    ----------
#    xdata : numpy.ndarray
#    ydata : numpy.ndarray
#    binsx : int
#        Number of bins along x dimension
#    binsy : int
#        Number of bins along y dimension
#    ax : matplotlib.Axes (optional)
#        If supplied, plot the contour to this axis. Otherwise, open a new figure
#    contour_kwargs : dict
#        kwargs to be passed to pyplot.contour()
#    """
#     # nbins_x = len(binsx) - 1
#     # nbins_y = len(binsy) - 1
#     import scipy.optimize as so
#
#     H, xedges, yedges = np.histogram2d(
#         xdata, ydata, bins=[binsx, binsy], normed=True)
#     x_bin_sizes = (xedges[1:] - xedges[:-1])
#     y_bin_sizes = (yedges[1:] - yedges[:-1])
#
#     pdf = (H * (x_bin_sizes * y_bin_sizes))
#
#     levels = [so.brentq(find_confidence_interval, 0., 1.,
#                         args=(pdf, prc)) for prc in levels_prc]
#
#     X, Y = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
#     Z = pdf.T
#
#     if ax == None:
#         contour = plt.contour(X, Y, Z, levels=levels, origin="lower",
#                               **contour_kwargs)
#         out = contour
#         if fill == True:
#             contourf = plt.contourf(X, Y, Z, levels=levels, origin="lower",
#                                     **contour_kwargs)
#             out = contour, contourf
#     else:
#         contour = ax.contour(X, Y, Z, levels=levels, origin="lower",
#                              **contour_kwargs)
#         out = contour
#         if fill == True:
#             contourf = ax.contourf(X, Y, Z, levels=levels, origin="lower",
#                                    **contour_kwargs)
#             out = contour, contourf
#
#     return out
#


def find_confidence_interval(hist_pdf, prc):
    """Find confidence interval from PDF.

    Parameters
    ----------
    hist_pdf : numpy.ndarray
        PDF data
    prc : float
        confidence level

    Returns
    -------
    out : float
        confidence interval
    """
    sorted_pdf = np.sort(hist_pdf.ravel())
    return np.interp(prc, np.linspace(0, 1, len(sorted_pdf)), sorted_pdf)


def density_contour(xdata, ydata, binsx, binsy, ax=None, range=None, fill=False, levels_prc=[.68, .95, .99], **contour_kwargs):
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
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()

    Returns
    -------
    out : tuple
        A tuple containing the contour plot(s) and the calculated contour levels.
    """
    H, xedges, yedges = np.histogram2d(
        xdata, ydata, bins=[binsx, binsy], density=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1])
    y_bin_sizes = (yedges[1:] - yedges[:-1])

    pdf = (H * (x_bin_sizes * y_bin_sizes))

    levels = [find_confidence_interval(pdf, prc) for prc in levels_prc]

    X, Y = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
    Z = pdf.T

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


def contour_pdf(x_axis, y_axis, ax=None, nbins=10, percent=[10],
                colors=['b'], pdf_resample=100, verbose=False):
    '''
    contornos para percentis tirei deste site:
    http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    '''
    __name__ = 'contour_pdf'
    call_logger(__name__)
    logger = logging.getLogger(__name__)

    logger.info(
        " - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Calculating PDF for %i bins and resolution of %i" % (nbins, pdf_resample)]))
    xmin, xmax = min(x_axis), max(x_axis)
    ymin, ymax = min(y_axis), max(y_axis)
    xf = np.transpose((x_axis, y_axis))
    pdf = scipy.stats.kde.gaussian_kde(xf.T)

    percentiles = np.array(percent)
    if len(percentiles) != len(colors):
        logger.warning(
            " - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Number of percentiles differs from the number of colors. Setting colors to None."]))
        colors = [None] * len(percentiles)

    if verbose:
        logger.info(
            " - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Calculating scores for percentiles %s" % repr(percentiles)]))
    scores = scoreatpercentile(pdf(pdf.resample(pdf_resample)), percentiles)

    if verbose:
        logger.info(
            " - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Calculating grid values for %i bins" % nbins]))
    q, w = np.meshgrid(np.linspace(xmin, xmax, nbins),
                       np.linspace(ymin, ymax, nbins))
    r = pdf([q.flatten(), w.flatten()])
    r.shape = q.shape

    if verbose:
        logger.info(
            " - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Creating contour plots..."]))
    if ax is None:
        ax = plt.gca()

    return ax.contour(np.linspace(xmin, xmax, nbins),
                      np.linspace(ymin, ymax, nbins),
                      r, scores, linewidths=1.5, colors=colors)
