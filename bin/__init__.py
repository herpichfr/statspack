#!/usr/bin/env python3
# statspack/__init__.py

# Import sub-modules or functions
from .binning import binning
from .find_confidence_interval import find_confidence_interval
from .density_contour import density_contour
from .contour_pdf import contour_pdf

# Define what names are imported with 'from statspack import *'
__all__ = ['binning', 'find_confidence_interval',
           'density_contour', 'contour_pdf']

# Optional initializations or configurations
# For example, you can set up logging here if needed
import logging

logging.basicConfig(level=logging.INFO)
