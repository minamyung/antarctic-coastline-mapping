#########################################
# Original Author: Anastasija Jadrevska #
# Modifying Author: Manon Myung         #
#########################################


# this code runs the process for a single file

input_location = "\\Users\myung\Documents\CSC8099\Input\\"
output_location = "\\Users\myung\Documents\CSC8099\Output\\"

n = 0

input_file = input_location + 'Output' + n + '.tif'
input_file_line = input_location + 'Output' + n + '_line.tif'
output_file = output_location + 'Vector' + n + '.shp'

# Thins lines from input file
processing.run("grass7:r.thin", {'input':input_file,'iterations':200,'output':input_file_line,'GRASS_REGION_PARAMETER':None,
                                 'GRASS_REGION_CELLSIZE_PARAMETER':0,'GRASS_RASTER_FORMAT_OPT':'','GRASS_RASTER_FORMAT_META':''})

# Converts raster data to vector data
processing.run("grass7:r.to.vect", {'input':nfn,'type':0,'column':'value','-s':False,'-v':False,'-z':False,'-b':False,'-t':False,'output':output_file,
                                    'GRASS_REGION_PARAMETER':None,'GRASS_REGION_CELLSIZE_PARAMETER':0,'GRASS_OUTPUT_TYPE_PARAMETER':2,'GRASS_VECTOR_DSCO':'','GRASS_VECTOR_LCO':'','GRASS_VECTOR_EXPORT_NOCAT':False})