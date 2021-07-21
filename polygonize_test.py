import time
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, ogr
from osgeo import osr
from scipy import ndimage
import subprocess
import geopandas as gpd
from shapely.geometry import Point, Polygon, box

startTime = time.time()

# location of source images
filepath = 'C:/Users/myung/Documents/CSC8099/Data/Coastline_images/'

# location of polygon data
polygon_filepath = 'C:/Users/myung/Documents/CSC8099/Data/polygons/'
# location to save processed images
nfnb= 'C:/Users/myung/Documents/CSC8099/Data/polygons/'


images = ['S1B_IW_GRDH_1SSH_20210711T043551_B06E_S_1.tif']
# for name in glob.glob("C:/Users/myung/Documents/CSC8099/Data/Coastline_images/*.tif"):
#     trunc_name = str(name).split('\\')[-1]
#     images.append(trunc_name) # Save the truncated (w/o path) image file names to make naming the result products easier

REFERENCE_POLYGON_COASTLINE_PATH = 'C:/Users/myung/Documents/CSC8099/Data/add_coastline_high_res_polygon_v7_4_dissolved.shp'


def read_img(filename):
    # read an image
    img_r = cv2.imread(filename, 0)
    img = img_r.astype(np.uint8)

    # read it as a GeoTiff to collect geodata
    img_m = gdal.Open(filename)
    img_array = img_m.ReadAsArray()

    # create a NO DATA mask
    bool_mask = (img_array != 0)
    img_mask = bool_mask.astype(np.uint8)
    return img, img_mask, img_m

def b_filter(img_in):
    blur = cv2.bilateralFilter(img_in, 9, 75, 75)
    return blur

def get_binary(blur, threshold=200, max_val=300):
    # q: Diameter of each pixel neighborhood that is used during filtering
    (thresh, binary) = cv2.threshold(blur, threshold, max_val, (cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    return binary

def delete_b(img, min_size_fraction):
    # split into components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)

    # remove background
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # find the largest component
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_size = sizes[i]

    # set to optimal min size
    min_size = max_size * min_size_fraction

    img2 = np.zeros(output.shape)
    # for every component in the image check if it larger than set minimum size
    for i in range(0, nb_components):
        if sizes[i] > min_size:
            img2[output == i + 1] = 1
    return img2

def extract_polygons(img, geo_file, mask):
    # Make an output layer to store polygon data
    # From https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    dst_layername = "polygon_layer"
    dst_drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = dst_drv.CreateDataSource(nfnb + dst_layername + '.shp')
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=None)
    
    # Create raster from current array
    # From https://stackoverflow.com/questions/37648439/simplest-way-to-save-array-into-raster-file-in-python
    src_drv = gdal.GetDriverByName("GTiff")
    src_layername = "source_layer"
    raster = src_drv.Create(nfnb + src_layername + ".tif", xsize=geo_file.RasterXSize, ysize=geo_file.RasterYSize, bands=1,
                                eType=gdal.GDT_UInt16)
    raster.GetRasterBand(1).WriteArray(img)
    geotransform = geo_file.GetGeoTransform()
    projection = geo_file.GetProjection()
    raster.SetGeoTransform(geotransform)
    raster.SetProjection(projection)
    raster.FlushCache()
    raster = None
    src = gdal.Open(nfnb + src_layername + ".tif")
    # Extract polygons
    gdal.Polygonize(src.GetRasterBand(1), geo_file.GetRasterBand(1), dst_layer, -1, [], callback=None)

def polygon_filtering(image_name):
    # NOTE: this currently does NOT dissolve the reference coastline, so all the internal features and lines are still present.

    # Clip ref to extent of the image
    # Reference coastline (polygon)
    ref = gpd.GeoSeries.from_file('C:/Users/myung/Documents/CSC8099/Data/add_coastline_high_res_polygon_v7_4/add_coastline_high_res_polygon_v7_4.shp')
    # Image (tif)
    border = gdal.Open('C:/Users/myung/Documents/CSC8099/Data/Coastline_images/S1B_IW_GRDH_1SSH_20210711T043551_B06E_S_1.tif')
    # Get bounds in coordinates
    geoTransform = border.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * border.RasterXSize
    miny = maxy + geoTransform[5] * border.RasterYSize
    border = None # close file
    # Draw a bounding box from coordinates (using shapely.geometry.box)
    bounding_box = box(minx, miny, maxx, maxy)
    # Make a geopandas-compatible dataframe (see pandas)
    df = {'name': ['bounding box'], 'geometry': [bounding_box]}
    # Make a GeoDataFrame which contains the bounding box in the correct coordinates and crs
    gdf = gpd.GeoDataFrame(df, crs='epsg:3031')
    # Save bounding box to shapefile if needed
    # gdf.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/bounding_box.shp')
    # Clip ref coastline to bounding box
    clipped = gpd.clip(ref, gdf)
    # Save clipped ref polygon coastline if needed
    # clipped.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/clipped.shp')

# Testing polygon_filtering
polygon_filtering(images[0]) 

def remove_mask(img, img_mask):
    img2 = cv2.bitwise_xor(img,img_mask)
    return img2

def remove_border(img, boundary):
    img2 = cv2.bitwise_and(img, img, mask=boundary)
    return img2

i = 0 # Count the number of images processed

# for image_name in images:
#     i += 1
#     total = len(images)

#     filename = filepath + image_name

#     print(str(i) + "/" + str(total))
#     print('Processing ' + image_name)

#     (image, mask, geo_file) = read_img(filename)
#     blur = b_filter(image).astype(np.uint8)
#     # plt.imshow(blur)
#     # plt.show()
#     binary = get_binary(blur).astype(np.uint8)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))

#     binary_w = delete_b(binary, 0.05).astype(np.uint8)
#     m_close_binary_w = cv2.morphologyEx(binary_w, cv2.MORPH_CLOSE, kernel)
#     # plt.imshow(m_close_binary_w)
#     # plt.show()
#     new_b = remove_mask(m_close_binary_w, mask).astype(np.uint8)
#     # plt.imshow(new_b)
#     # plt.show()
#     # Here, extract polygons and remove those which are inside the ref. coastline as internal features
    
#     # m_close_new_b = cv2.morphologyEx(new_b, cv2.MORPH_CLOSE, kernel)
#     #m_open_new_b = cv2.morphologyEx(new_b, cv2.MORPH_OPEN, kernel)
#     # plt.imshow(m_close_new_b)
#     # plt.show()
#     # plt.imshow(m_open_new_b)
#     # plt.show()
#     extract_polygons(new_b, geo_file, mask)
#     # new_clean = extract_polygons(new_b, 0.05).astype(np.uint8)
#     # plt.imshow(new_clean)
#     # plt.show()
#     polygon_filtering(image_name)


