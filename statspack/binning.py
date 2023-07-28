#!/usr/bin/env python3
# package with modules for statistic visualization
# Fabio R. Herpich 2023-07-26 CASU/IoA Cambridge
# All rights reserved (see LICENSE file)

import numpy as np
from datetime import datetime
import logging
import colorlog


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


def binning(x, y, z, nbins=10, xlim=(None, None), ylim=(None, None), verbose=False):
    """Binning data for contour plots.

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
