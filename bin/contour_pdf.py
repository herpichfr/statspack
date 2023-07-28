#!/usr/bin/env python3
# package with modules for statistic visualization
# Fabio R. Herpich 2023-07-26 CASU/IoA Cambridge
# All rights reserved (see LICENSE file)

import numpy as np
import scipy.stats
from datetime import datetime
import logging
import colorlog
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


def contour_pdf(x_axis, y_axis, ax=None, nbins=10, percent=[10],
                colors=['b'], pdf_resample=100, verbose=False):
    '''
    Create contours for a given set of percentiles over a 2D representation of the data.

    Parameters
    ----------
    x_axis : numpy.ndarray
        x-axis data
    y_axis : numpy.ndarray
        y-axis data
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    nbins : int
        Number of bins along each axis
    percent : list
        List of percentiles to plot. Default is [10]
    colors : list
        List of colors to use for each percentile. Default is ['b']
    pdf_resample : int
        Number of points to use when calculating the PDF. Default is 100
    verbose : bool
        verbose

    Returns
    -------
    contours: matplotlib.contour.QuadContourSet
        A QuadContourSet of the contours
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
