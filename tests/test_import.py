# Tests the required imports for the sake of my sanity
# Recommended: use conda and conda-forge to install the necessary packages
# Use a conda env (in this case acm) to run the code
# Make sure that VScode recognises conda and the env is correctly activated!!!!!!!!!!!!!!

from osgeo import gdal
import cv2
import numpy as np
import scipy
import matplotlib
import tifffile as tifi


assert(gdal.__version__ is not None), "GDAL import failed"
assert(cv2.__version__ is not None), "cv2 import failed"
assert(np.__version__ is not None), "numpy import failed"
assert(scipy.__version__ is not None), "scipy import failed"
assert(matplotlib.__version__ is not None), "matplotlib import failed"
assert(tifi.__version__ is not None), "tifffile import failed"
print("Imports OK.")