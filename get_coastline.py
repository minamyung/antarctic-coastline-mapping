#########################################
# Original Author: Anastasija Jadrevska #
# Modifying Author: Manon Myung         #
#########################################



import glob
import sys
import time
from collections import defaultdict

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
import rasterio.mask
from affine import Affine
from osgeo import gdal
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon, shape

start_time = time.time()


# Note: all lengths/areas are in m/m^2 unless otherwise stated

# Location of source images
SRC_FILEPATH = 'C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/'

# Location to save outputs
DST_FILEPATH = 'C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/output/'

# Reference coastline for polygon masking
REF_PATH = 'C:/Users/myung/Documents/CSC8099/Data/add_coastline_high_res_polygon_v7_4/add_coastline_high_res_polygon_v7_4.shp'
REF_GDF = gpd.GeoDataFrame.from_file(REF_PATH)

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)) # Kernel element for morphological open/close - smoothing
MIN_AREA = 1000000000  # Minimum area threshold in m^2 (1000km^2) - any components under this size will be considered to be noise

def read_img(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

    # read it as a GeoTiff to collect geodata
    geo_file = gdal.Open(filename)
    img_array = geo_file.GetRasterBand(1).ReadAsArray()

    # create a NO DATA mask
    bool_mask = (img_array != 0)
    img_mask = bool_mask.astype(np.uint8)


    # Get raster transform from geo_file
    transform = Affine.from_gdal(*geo_file.GetGeoTransform())
    # 2d transform method from
    # https://gis.stackexchange.com/questions/380357/affine-tranformation-matrix-shapely-asks-6-coefficients-but-rasterio-delivers
    two_d_transform = [
        elem for tuple in transform.column_vectors for elem in tuple
    ]

    return img, img_mask, transform, two_d_transform


def bilateral_filter(img):
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def get_binary(img):
    # Takes a raster array (needs to be np.uint8) and returns its otsu threshold + the result of the thresholding
    # Calculated by disregarding black areas (border)

    # Remove the cells which have a value of 0 (border)
    bool_index = (img != 0)
    # Add them to an array
    temp_img = np.extract(bool_index, img)
    # Calculate Otsu threshold from the pixels which have a non-zero value
    thresh_val, temp = cv2.threshold(temp_img, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Apply thresholding to raster array with the threshold value calculated above
    ret, thresh_img = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    return thresh_img


def filter_components(img, transform):
    # Get pixel sizes
    dx = transform.column_vectors[0][0]
    dy = -transform.column_vectors[1][1]

    # Scale the min component area using pixel size
    scaled_min_area = MIN_AREA / (dx * dy)

    # split into components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=4)

    # remove background
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # Make an array to hold filtered image
    filtered_img = np.zeros(output.shape)

    for i in range(0, nb_components):
        if sizes[i] > scaled_min_area:
            filtered_img[output == i + 1] = 1
    return filtered_img

def mask_invert(img, img_mask):
    img2 = cv2.bitwise_xor(img, img_mask)
    return img2


def get_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contours = map(np.squeeze, contours)
    polygons = map(Polygon, contours)
    #cv2.drawContours(img, contours, -1, (255, 0, 0), 250)
    return img


def mask_to_polygons(mask, clipped_inner, two_d_transform, buffer=0, min_area=10):
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8), cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return MultiPolygon()

    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            # If the contour has child contours, add their index to child_contours
            child_contours.add(idx)
            # And append the contour to the list of its parent's child contours
            cnt_children[parent_idx].append(contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(shell=cnt[:, 0, :],
                           holes=[
                               c[:, 0, :] for c in cnt_children.get(idx, [])
                               if cv2.contourArea(c) >= min_area
                           ])
            all_polygons.append(poly.buffer(buffer))

    # Create geoDataFrame
    polygons_names = []

    for poly in all_polygons:
        polygons_names.append('contour polygon')

    df_polygons = {'name': polygons_names, 'geometry': all_polygons}
    gdf_polygons = gpd.GeoDataFrame(
        df_polygons, crs='epsg:3031').affine_transform(two_d_transform)
    #print(gdf_polygons)
    #gdf_polygons.to_file('C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/all_ocean_new.shp')

    # retain only land polygons (remove misclassified ocean areas)
    inner_polygons = []
    for ref_polygon in clipped_inner['geometry']:
        for i in range(len(gdf_polygons)):
            polygon = gdf_polygons[i]
            intersect = ref_polygon.intersection(polygon)
            intersect_area = intersect.area
            intersect_percentage = intersect_area / polygon.area
            if intersect_percentage > 0.9:
                inner_polygons.append(all_polygons[i])

    return inner_polygons


def get_bounding_box(img_mask, transform):
    # Returns a polygon of the size of the image bounds

    # Initialise mask_polygon
    mask_polygon = None
    # Go through all the shapes extracted from the img_mask array
    for vec in rasterio.features.shapes(img_mask, transform=transform):
        if (vec[1] == 1
            ):  # The inner box of the img_mask is where the value is 1
            mask_polygon = shape(vec[0])  # Save as a shape
    # Make the dataframe (see pandas)
    df_inner = {'name': ['inner bounding box'], 'geometry': [mask_polygon]}
    #df_inner = {'geometry': mask_polygon}
    # Make a GeoDataFrame with the right crs
    gdf_inner = gpd.GeoDataFrame(df_inner, crs='epsg:3031')
    # Save to shapefile if necessary
    # gdf_inner.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/bounding_box_inner.shp')
    return gdf_inner


def clip_ref_polygon_inner(gdf_inner):
    # Clip reference polygon data to inner border of the satellite image(w/o nodata areas)
    # Clip ref coastline to bounding box
    clipped_inner = gpd.clip(REF_GDF, gdf_inner)
    # Save to shapefile if necessary
    # clipped_inner.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/clipped_inner.shp')
    # Dissolve to remove internal features and only get the outermost coastline
    clipped_inner_dissolved = clipped_inner.dissolve()
    # Save to shapefile if necessary
    # clipped_inner_dissolved.to_file('C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/clipped_inner_dissolved.shp')
    return clipped_inner_dissolved


def polygons_to_shpfile(polygons, two_d_transform, dst_filename):
    polygons_transformed = []
    polygons_names = []

    for poly in polygons:
        polygons_transformed.append(affine_transform(poly, two_d_transform))
        polygons_names.append('contour polygon')

    df_polygons = {'name': polygons_names, 'geometry': polygons_transformed}
    gdf_polygons = gpd.GeoDataFrame(df_polygons, crs='epsg:3031')
    gdf_dissolved = gdf_polygons.dissolve()
    gdf_dissolved.to_file(dst_filename + '.shp')
    return gdf_dissolved

def get_change(dissolved_polygons, clipped_inner_dissolved):
    change_polygon = gpd.overlay(dissolved_polygons, clipped_inner_dissolved, how='difference')
    change_area = 0
    for geom in change_polygon['geometry']:
        change_area += geom.area

    return change_polygon, change_area

# def save_gtiff(img, nfn):
#     print(nfn)
#     driver_tiff = gdal.GetDriverByName("GTiff")

#     # create GeoTiff
#     nds = driver_tiff.Create(nfn, xsize=geo_file.RasterXSize, ysize=geo_file.RasterYSize, bands=1,
#                                 eType=gdal.GDT_UInt16)
#     nds.SetGeoTransform(geo_file.GetGeoTransform())
#     nds.SetProjection(geo_file.GetProjection())

#     # copy output to the created file
#     bandn = nds.GetRasterBand(1).ReadAsArray()
#     bandn = img
#     nds.GetRasterBand(1).WriteArray(bandn)

#     # set no data to remove background
#     nds.GetRasterBand(1).SetNoDataValue(0)
#     nds = None



# Add all images to process to list
images = []
for name in glob.glob(SRC_FILEPATH + "*.tif"):
    trunc_name = str(name).split('\\')[-1]
    images.append(
        trunc_name
    )  # Save the truncated (w/o path) image file names to make naming the result products easier


i = 0  # Count the number of images processed

for image_name in images:
    i += 1
    total = len(images)

    filename = SRC_FILEPATH + image_name

    # set the new file destination (same as source image, but different folder)
    dst_filename = DST_FILEPATH + str(image_name).split('.')[0]

    print(str(i) + "/" + str(total))
    print('Processing ' + image_name)

    (img, img_mask, transform, two_d_transform) = read_img(filename)

    # Apply bilateral filter (blurring)
    blur = bilateral_filter(img).astype(np.uint8)
    # plt.imshow(blur)
    # plt.show()

    # Get binary image
    binary = get_binary(blur).astype(np.uint8)
    # plt.imshow(binary)
    # plt.show()

    # Remove noise components in ocean
    binary_w = filter_components(binary, transform).astype(np.uint8)
    # plt.imshow(binary_w)
    # plt.show()

    # Morphological close
    m_close_binary_w = cv2.morphologyEx(binary_w, cv2.MORPH_CLOSE, KERNEL).astype(np.uint8)
    # plt.imshow(m_close_binary_w)
    # plt.show()

    # Invert
    new_b = mask_invert(m_close_binary_w, img_mask).astype(np.uint8)
    # plt.imshow(new_b)
    # plt.show()

    # Morphological open
    m_open_new_b = cv2.morphologyEx(new_b, cv2.MORPH_OPEN,
                                    KERNEL).astype(np.uint8)
    # plt.imshow(m_open_new_b)
    # plt.show()

    # Remove noise components inside coast
    new_clean = filter_components(m_open_new_b, transform).astype(np.uint8)
    # plt.imshow(new_clean)
    # plt.show()

    # Revert back
    new_clean_invert = mask_invert(new_clean, img_mask).astype(np.uint8)
    # plt.imshow(new_clean_invert)
    # plt.show()

    # Get a bounding box of the image extent
    gdf_inner = get_bounding_box(img_mask, transform)

    # Use bounding box to clip the reference coastline polygons, and dissolve them
    clipped_inner_dissolved = clip_ref_polygon_inner(gdf_inner)

    # Get the polygons of the areas classified as land
    polygons = mask_to_polygons(new_clean_invert, clipped_inner_dissolved,
                                two_d_transform)

    # Get the polygons of land/inner features misclassified as ocean
    inner_polygons = mask_to_polygons(new_clean, clipped_inner_dissolved, two_d_transform, 10)

    # Gather all resulting polygons
    polygons.extend(inner_polygons)

    # Dissolve polygons to have a unified coastline polygon, and save to shapefile
    dissolved = polygons_to_shpfile(polygons, two_d_transform, dst_filename)

    # Change detection using the change polygon method
    positive_change_polygons, positive_change_area = get_change(dissolved, clipped_inner_dissolved)
    negative_change_polygons, negative_change_area = get_change(clipped_inner_dissolved, dissolved)
    
    # Calculate net change
    net_change_area = positive_change_area - negative_change_area
    
    # Save change polygons
    positive_change_polygons.to_file(dst_filename + '_pos_change_' + '.shp')
    negative_change_polygons.to_file(dst_filename + '_neg_change_' + '.shp')
    
    print('Positive change: '+ str(positive_change_area/1000000) + ' km^2')
    print('Negative change: ' + str(negative_change_area/1000000) + ' km^2')
    print('Net change: ' + str(net_change_area/1000000) + ' km^2')
    
execution_time = (time.time() - start_time)
print("Finished processing images.")
print("Execution time: " + str(execution_time))
print("Average time per image: " + str(execution_time/i))
