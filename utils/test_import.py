# Tests the required imports
# Recommended: use conda and conda-forge to install the necessary packages
# Use a conda env to run the code


from osgeo import gdal
import cv2
import numpy as np
import scipy
import matplotlib
import geopandas as gpd
import rasterio
import affine
import shapely

assert(gdal.__version__ is not None), "GDAL import failed"
assert(cv2.__version__ is not None), "cv2 import failed"
assert(np.__version__ is not None), "numpy import failed"
assert(scipy.__version__ is not None), "scipy import failed"
assert(matplotlib.__version__ is not None), "matplotlib import failed"
assert(gpd.__version__ is not None), "matplotlib import failed"
assert(rasterio.__version__ is not None), "matplotlib import failed"
assert(affine.__version__ is not None), "matplotlib import failed"
assert(shapely.__version__ is not None), "tifffile import failed"
print("Imports OK.")