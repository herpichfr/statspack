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
