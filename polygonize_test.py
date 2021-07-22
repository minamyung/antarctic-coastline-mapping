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
from shapely.geometry import Point, Polygon, box, shape
import rasterio.features
from affine import Affine

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

REF_PATH = 'C:/Users/myung/Documents/CSC8099/Data/add_coastline_high_res_polygon_v7_4/add_coastline_high_res_polygon_v7_4.shp'
REF_GS = gpd.GeoSeries.from_file(REF_PATH)
REF_GDF = gpd.GeoDataFrame.from_file(REF_PATH)
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

def extract_polygons(img, img_mask, img_m):
    # Try again with rasterio instead of gdal
    transform = Affine.from_gdal(*img_m.GetGeoTransform())
    # Initialise mask_polygon
    polygons = []
    polygons_names = []
    # Go through all the shapes extracted from the img_mask array 
    for vec in rasterio.features.shapes(img, transform=transform): 
        polygons.append(shape(vec[0])) # Save as a shape
        polygons_names.append('extracted polygons')
    
    df_polygons = {'name': polygons_names, 'geometry': polygons}
    gdf_polygons = gpd.GeoDataFrame(df_polygons, crs='epsg:3031')

    # for getting rid of the no data area
    mask_polygon = None
    # Go through all the shapes extracted from the img_mask array 
    for vec in rasterio.features.shapes(img_mask, transform=transform): 
        if(vec[1] == 1): # The inner box of the img_mask is where the value is 1
            mask_polygon = shape(vec[0]) # Save as a shape
    # Make the dataframe (see pandas)
    df_inner = {'name': ['inner bounding box'], 'geometry': [mask_polygon]}
    # Make a GeoDataFrame with the right crs
    gdf_inner = gpd.GeoDataFrame(df_inner, crs='epsg:3031')

    clip_polygons = gpd.clip(gdf_polygons, gdf_inner)
    clip_polygons.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/clipped_polygons.shp')
    return clip_polygons

def extract_polygons_old(img, img_mask, img_m):
    # Make an output layer to store polygon data
    # From https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    dst_layername = "polygon_layer_0.05area_nomask_open10_1"
    dst_drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = dst_drv.CreateDataSource(nfnb + dst_layername + '.shp')
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=None)
    
    
    # Create raster from current array
    # From https://stackoverflow.com/questions/37648439/simplest-way-to-save-array-into-raster-file-in-python
    src_drv = gdal.GetDriverByName("GTiff")
    src_layername = "source_layer_0.05area_nomask_open10_1"
    raster = src_drv.Create(nfnb + src_layername + ".tif", xsize=img_m.RasterXSize, ysize=img_m.RasterYSize, bands=1,
                                eType=gdal.GDT_UInt16)
    raster.GetRasterBand(1).WriteArray(img)
    geotransform = img_m.GetGeoTransform()
    projection = img_m.GetProjection()

    raster.SetGeoTransform(geotransform)
    raster.SetProjection(projection)
    raster.FlushCache()
    raster = None
    src = gdal.Open(nfnb + src_layername + ".tif")
    # Extract polygons
    # gdal.Polygonize(src.GetRasterBand(1), img_m.GetRasterBand(1), dst_layer, -1, [], callback=None)
    # Check to see if the mask was doing anything
    gdal.Polygonize(src.GetRasterBand(1), None, dst_layer, -1, [], callback=None)
    # This is to make the file available later on!!!
    dst_ds.FlushCache()
    dst_ds = None
    # Clip to size
    # Open dst_layer with geopandas
    layer_gdf = gpd.GeoDataFrame.from_file('C:/Users/myung/Documents/CSC8099/Data/polygons/polygon_layer_0.05area_nomask_open10_1.shp')
    layer_gdf.crs = 'epsg:3031'
    layer_gdf.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/layer_gdf.shp')
    transform = Affine.from_gdal(*img_m.GetGeoTransform())
    # Initialise mask_polygon
    mask_polygon = None
    # Go through all the shapes extracted from the img_mask array 
    for vec in rasterio.features.shapes(img_mask, transform=transform): 
        if(vec[1] == 1): # The inner box of the img_mask is where the value is 1
            mask_polygon = shape(vec[0]) # Save as a shape
    # Make the dataframe (see pandas)
    df_inner = {'name': ['inner bounding box'], 'geometry': [mask_polygon]}
    # Make a GeoDataFrame with the right crs
    gdf_inner = gpd.GeoDataFrame(df_inner, crs='epsg:3031')
    # Save to shapefile if necessary
    # gdf_inner.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/bounding_box_inner_01.shp')
    # Clip polygonized data to bounding box
    clipped_inner = gpd.clip(layer_gdf, gdf_inner)
    # Save to shapefile if necessary
    clipped_inner.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/polygonize_clipped_inner.shp')
    # Dissolve to remove internal features and only get the outermost coastline
    #clipped_inner_dissolved = clipped_inner.dissolve()
    # Save to shapefile if necessary
    #clipped_inner_dissolved.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/polygonize_clipped_inner_dissolved_01.shp')
    #return clipped_inner_dissolved

def clip_ref_polygon(img_m):
    # NOTE: this currently does NOT dissolve the reference coastline, so all the internal features and lines are still present.

    # Clip ref to extent of the image
    # Get bounds in coordinates (Affine transform)
    geoTransform = img_m.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * img_m.RasterXSize
    miny = maxy + geoTransform[5] * img_m.RasterYSize
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
    clipped = gpd.clip(REF_GS, gdf)
    # Save clipped ref polygon coastline if needed
    # clipped.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/clipped.shp')
    return clipped


def clip_ref_polygon_inner(img_mask, img_m):
    # NOTE: this currently does NOT dissolve the reference coastline, so all the internal features and lines are still present.
    # Now try with the inner border(w/o nodata areas)
    # Get Affine transform of original image
    transform = Affine.from_gdal(*img_m.GetGeoTransform())
    # Initialise mask_polygon
    mask_polygon = None
    # Go through all the shapes extracted from the img_mask array 
    for vec in rasterio.features.shapes(img_mask, transform=transform): 
        if(vec[1] == 1): # The inner box of the img_mask is where the value is 1
            mask_polygon = shape(vec[0]) # Save as a shape
    # Make the dataframe (see pandas)
    df_inner = {'name': ['inner bounding box'], 'geometry': [mask_polygon]}
    # Make a GeoDataFrame with the right crs
    gdf_inner = gpd.GeoDataFrame(df_inner, crs='epsg:3031')
    # Save to shapefile if necessary
    # gdf_inner.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/bounding_box_inner.shp')
    # Clip ref coastline to bounding box
    clipped_inner = gpd.clip(REF_GDF, gdf_inner)
    # Save to shapefile if necessary
    # clipped_inner.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/clipped_inner.shp')
    # Dissolve to remove internal features and only get the outermost coastline
    clipped_inner_dissolved = clipped_inner.dissolve()
    # Save to shapefile if necessary
    # clipped_inner_dissolved.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/clipped_inner_dissolved.shp')
    return clipped_inner_dissolved

def polygon_filtering(extracted_polygons, clipped_inner):
    # extracted_polygons is a layer
    filtered_polygons = []
    for polygon in extracted_polygons:
        if not clipped_inner.contains(polygon):
            filtered_polygons.append(polygon)
    return filtered_polygons



# Testing clip_ref_polygon and clip_ref_polygon_inner

# img, img_mask, img_m = read_img(filepath + images[0])
# clip_ref_polygon_inner(img_mask, img_m) 

def remove_mask(img, img_mask):
    img2 = cv2.bitwise_xor(img,img_mask)
    return img2

def remove_border(img, boundary):
    img2 = cv2.bitwise_and(img, img, mask=boundary)
    return img2

i = 0 # Count the number of images processed

for image_name in images:
    i += 1
    total = len(images)

    filename = filepath + image_name

    print(str(i) + "/" + str(total))
    print('Processing ' + image_name)

    (img, img_mask, img_m) = read_img(filename)
    blur = b_filter(img).astype(np.uint8)
    # plt.imshow(blur)
    # plt.show()
    binary = get_binary(blur).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))

    binary_w = delete_b(binary, 0.05).astype(np.uint8)
    m_close_binary_w = cv2.morphologyEx(binary_w, cv2.MORPH_CLOSE, kernel)
    # plt.imshow(m_close_binary_w)
    # plt.show()
    new_b = remove_mask(m_close_binary_w, img_mask).astype(np.uint8)
    # plt.imshow(new_b)
    # plt.show()
    # Here, extract polygons and remove those which are inside the ref. coastline as internal features
    
    # m_close_new_b = cv2.morphologyEx(new_b, cv2.MORPH_CLOSE, kernel)
    m_open_new_b = cv2.morphologyEx(new_b, cv2.MORPH_OPEN, kernel)
    # plt.imshow(m_close_new_b)
    # plt.show()
    # plt.imshow(m_open_new_b)
    # plt.show()
    extract_polygons(m_open_new_b, img_mask, img_m)
    # new_clean = extract_polygons(new_b, 0.05).astype(np.uint8)
    # plt.imshow(new_clean)
    # plt.show()
    # polygon_filtering(image_name)

