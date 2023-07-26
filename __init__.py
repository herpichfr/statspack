# statspack/__init__.py

# Import sub-modules or functions
from .bining import bining
from .confidence_interval import find_confidence_interval
from .density_contour import density_contour
from .contour_pdf import contour_pdf

# Define what names are imported with 'from statspack import *'
__all__ = ['bining', 'find_confidence_interval',
           'density_contour', 'contour_pdf']

# Optional initializations or configurations
# For example, you can set up logging here if needed
import logging

logging.basicConfig(level=logging.INFO)
