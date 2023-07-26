#!/bin/env python3
# package with modules for statistic visualization
# herpich 2022-12-20 fabiorafaelh@gmail.com

import numpy as np
import logging
import colorlog
from datetime import datetime
import scipy.stats
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


def bining(x, y, z, nbins=10, xlim=(None, None), ylim=(None, None), verbose=False):
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
    verbose : bool
        verbose

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
    __name__ = 'statspack.bining'
    call_logger(__name__, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Initializing bining"]))

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
    if verbose:
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Calculating bin centres"]))
    x_center = (xv[:-1] + xv[1:]) / 2
    y_center = (yv[:-1] + yv[1:]) / 2

    # Create a 2D array of bin indices for each data point
    bin_indices = x_bins * nbins + y_bins

    # Check for invalid bin indices and discard them
    valid_indices_mask = (bin_indices >= 0) & (bin_indices < nbins**2)
    x_bins = x_bins[valid_indices_mask]
    y_bins = y_bins[valid_indices_mask]
    z = z[valid_indices_mask]
    bin_indices = bin_indices[valid_indices_mask]

    # Initialize empty arrays for the binned data
    if verbose:
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Initializing empty arrays for the binned data"]))
    X = x_center.repeat(nbins)
    Y = np.tile(y_center, nbins)
    Z = np.full(nbins**2, np.nan)

    # Calculate sums for each bin using np.add.at
    if verbose:
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Calculating sums for each bin using np.add.at"]))
    np.add.at(Z, bin_indices, z)

    # Calculate the count of data points in each bin
    bin_counts = np.bincount(bin_indices, minlength=nbins**2)

    # Replace NaN values in Z with 0 to avoid division by zero
    Z[np.isnan(Z)] = 0

    # Calculate the mean instead of median using np.divide
    if verbose:
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Calculating the mean instead of median using np.divide"]))
    Z = np.divide(Z, bin_counts, out=np.zeros_like(Z), where=(bin_counts != 0))

    return X, Y, Z


def find_confidence_interval(hist_pdf, prc, verbose=False):
    """Find confidence interval from PDF.

    Parameters
    ----------
    hist_pdf : numpy.ndarray
        PDF data
    prc : float
        confidence level
    verbose : bool
        verbose

    Returns
    -------
    out : float
        confidence interval
    """
    if verbose:
        __name__ = 'statspack.find_confidence_interval'
        call_logger(__name__, level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Initializing find_confidence_interval"]))
    sorted_pdf = np.sort(hist_pdf.ravel())
    return np.interp(prc, np.linspace(0, 1, len(sorted_pdf)), sorted_pdf)


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
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()

    Returns
    -------
    out : tuple
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


def contour_pdf(x_axis, y_axis, ax=None, nbins=10, percent=[10],
                colors=['b'], pdf_resample=100, verbose=False):
    '''
    contornos para percentis tirei deste site:
    http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    '''
    __name__ = 'statspack.contour_pdf'
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
    scores = scipy.stats.scoreatpercentile(
        pdf(pdf.resample(pdf_resample)), percentiles)

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
