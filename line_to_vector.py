#########################################
# Original Author: Anastasija Jadrevska #
# Modifying Author: Manon Myung         #
#########################################



# Must run in qgis

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import scipy

input_location = "\\Users\myung\Documents\CSC8099\Data\Input\\"
output_location = "\\Users\myung\Documents\CSC8099\Data\Output\\"


images = []
for name in glob.glob(input_location):
    trunc_name = str(name).split('\\')[-1]
    images.append(trunc_name)

i = 0

for image_name in images:
    i += 1
    total = len(images)

    print(str(i) + "/" + str(total))
    print('Processing ' + image_name)


    input_file = input_location + image_name + '.tif'
    input_file_line = input_location + image_name + '_line.tif'
    output_file = output_location + image_name + '_vector.shp'

    # Thins lines from input file
    processing.run("grass7:r.thin", {'input':input_file,'iterations':200,'output':input_file_line,'GRASS_REGION_PARAMETER':None,
                                    'GRASS_REGION_CELLSIZE_PARAMETER':0,'GRASS_RASTER_FORMAT_OPT':'','GRASS_RASTER_FORMAT_META':''})

    # Converts raster data to vector data
    processing.run("grass7:r.to.vect", {'input':input_file_line,'type':0,'column':'value','-s':False,'-v':False,'-z':False,'-b':False,'-t':False,'output':output_file,
                                        'GRASS_REGION_PARAMETER':None,'GRASS_REGION_CELLSIZE_PARAMETER':0,'GRASS_OUTPUT_TYPE_PARAMETER':2,'GRASS_VECTOR_DSCO':'','GRASS_VECTOR_LCO':'','GRASS_VECTOR_EXPORT_NOCAT':False})