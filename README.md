# Antarctic coastline mapping
## Description
Source code for dissertation project which aims to automate the digitisation of the Antactic coastline using satellite images (namely Sentinel 1).
Builds on work done by Anastasija Jadrevska.

## Dependencies
Several Python packages are required to run the scripts. These are listed in the requirements.txt file, which can be used to easily install the relevant packages into a virtual environment using a Python package manager such as Anaconda.

## Running the main algorithm
To run the main module, ./get_coastline.py, the variables SRC_FILEPATH and DST_FILEPATH must be set to the relevant filepaths, the former being a folder containing some GeoTiff files, and the latter being the folder where the shapefile outputs will be saved. 

The REF_PATH variable must be set to a folder containing the high-resolution polygon reference coastline, available at https://data.bas.ac.uk/items/cdeb448d-10de-4e6e-b56b-6a16f7c59095/.

## Accepted image types
The images must be in .tif format, in the EPSG:3031/WGS 84 Antarctic Polar Stereographic projection. The data type must be unsigned 8-bit or 16-bit integer. If the image contains more than one band, only the first band will be used. The image is read in greyscale for processing.

Suitable Sentinel-1 SAR images can be downloaded from https://www.polarview.aq/antarctic by selecting the desired time frame, choosing an image on the map and selecting the GeoTiff option.

## Output
For each GeoTiff image found in the SRC_FILEPATH folder, three shapefiles will be generated and saved to the DST_FILEPATH: the detected coastline polygon, and the positive and negative change polygons. Additionally, an text file named output.txt will be created in the DST_FILEPATH folder, containing the calculated change areas for each image.

## Utility scripts
The ./utils/ folder contains some helper scripts:

- test_import.py can be used to test imports (some Python packages used are prone to causing problems if not installed properly).

- otsu_mask.py can be used to display raster histograms with and without zero-valued pixels and the resulting Otsu threshold values.

- extractor.bat can be used to batch-unzip the coastline image files downloaded from PolarView.

- get_reference_coastline_files.py can be used to batch-download a list of images from PolarView (although images are archived after some time).

- get_scihub_files.py retrieves files from the SciHub API.

## Acknowledgements
Many thanks to my supervisor, David Herbert, from Newcastle University, as well as Louise Ireland and Laura Gerrish from the British Antarctic Survey.

## Contact
Manon Myung:

m.myung2@ncl.ac.uk

myung.mina@gmail.com

Distributed under the MIT license (see LICENSE.md).
