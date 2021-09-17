#########################################
# Original Author: Anastasija Jadrevska #
# Modifying Author: Manon Myung         #
#########################################



import glob
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

#################################################

# INPUT SRC_FILEPATH, DST_FILEPATH AND REF_PATH #

# Location of source images
SRC_FILEPATH = 'C:/Users/myung/Documents/CSC8099/demo/'

# Location to save outputs
DST_FILEPATH = 'C:/Users/myung/Documents/CSC8099/demo/'

# Reference coastline for polygon masking
REF_PATH = 'C:/Users/myung/Documents/CSC8099/Data/add_coastline_high_res_polygon_v7_4/add_coastline_high_res_polygon_v7_4_dissolved_fixed_geoms.shp'

#################################################

REF_GDF = gpd.GeoDataFrame.from_file(REF_PATH)

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)) # Kernel element for morphological open/close - smoothing
MIN_AREA = 1000000000  # Minimum area threshold in m^2 (1000km^2) - any components under this size will be considered to be noise

def read_img(filename):
    # Open tif file
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
    # Apply bilateral filter
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def get_binary(img):
    # Takes a raster array (needs to be np.uint8) and performs Otsu thresholding
    # Calculated by disregarding black pixels (border)

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
    print(nb_components)
    for i in range(0, nb_components):
        if sizes[i] > scaled_min_area:
            filtered_img[output == i + 1] = 1
    return filtered_img

def mask_invert(img, img_mask):
    # Invert binary image while ignoring the no-data area (image border)
    img2 = cv2.bitwise_xor(img, img_mask)
    return img2


def mask_to_polygons(mask, clipped_inner, two_d_transform, buffer=0, min_area=10):
    # Filter polygons using the reference coastline

    # Adapted from Konstantin Lopuhin's code
    # https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly

    # Find contours
    contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8), cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return MultiPolygon()

    # Associate parent and child contours
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

    # Create polygons from contours while removing artifacts
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

    # Create GeoDataFrame
    polygons_names = []

    for poly in all_polygons:
        polygons_names.append('contour polygon')

    df_polygons = {'name': polygons_names, 'geometry': all_polygons}
    gdf_polygons = gpd.GeoDataFrame(
        df_polygons, crs='epsg:3031').affine_transform(two_d_transform)
    # gdf_polygons.to_file('C:/Users/myung/Documents/CSC8099/Example_problems/polygon_filtering_new/bad_filtered_' + str(buffer) + '.shp')

    # Remove misclassified components using the reference coastline polygon
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
            mask_polygon = shape(vec[0]).buffer(0)  # Save as a shape
    # Make the dataframe (see pandas)
    df_inner = {'name': ['inner bounding box'], 'geometry': [mask_polygon]}
    #df_inner = {'geometry': mask_polygon}
    # Make a GeoDataFrame with the right crs
    gdf_inner = gpd.GeoDataFrame(df_inner, crs='epsg:3031')
    # Save to shapefile if necessary
    # gdf_inner.to_file('C:/Users/myung/Documents/CSC8099/Data/polygons/bounding_box_inner.shp')
    # border_mask = gdf_inner.boundary.buffer(50)
    # border_mask.to_file('C:/Users/myung/Documents/CSC8099/Data/rn_final/no_border_change_polygon/border_mask.shp')
    return gdf_inner


def clip_ref_polygon_inner(gdf_inner):
    # Clip reference polygon data to inner border of the satellite image(w/o nodata areas)
    # Clip ref coastline to bounding box
    clipped_inner = gpd.clip(REF_GDF, gdf_inner)
    return clipped_inner


def polygons_to_shpfile(polygons, two_d_transform, dst_filename):
    # Save polygons to shapefile with the right transform
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


def get_change(geom1, geom2):
    # Calculate difference between two GeoDataFrames
    change_polygon = gpd.overlay(geom1, geom2, how='difference')
    
    change_area = 0
    for geom in change_polygon['geometry']:
        change_area += geom.area

    return change_polygon, change_area

# Add all images to process to list
images = []
for name in glob.glob(SRC_FILEPATH + "*.tif"):
    trunc_name = str(name).split('\\')[-1]
    images.append(
        trunc_name
    )  # Save the truncated (w/o path) image file names to make naming the result products easier

with open(DST_FILEPATH + 'output.txt', 'a') as f:

    i = 0  # Count the number of images processed

    for image_name in images:
        i += 1
        total = len(images)

        filename = SRC_FILEPATH + image_name

        # set the new file destination (same as source image, but different folder)
        dst_filename = DST_FILEPATH + str(image_name).split('.')[0]

        print(str(i) + "/" + str(total))
        print('Processing ' + image_name)
        f.write(str(i) + "/" + str(total) + '\n')
        f.write('Processing ' + image_name + '\n')

        (img, img_mask, transform, two_d_transform) = read_img(filename)

        # Apply bilateral filter (blurring)
        blur = bilateral_filter(img).astype(np.uint8)
        print('blur done')
        plt.imshow(blur)
        plt.show()

        # Get binary image
        binary = get_binary(blur).astype(np.uint8)
        print('binary done')
        

        # Remove noise components in ocean
        binary_w = filter_components(binary, transform).astype(np.uint8)
        print('binary_w done')


        # Morphological close
        m_close_binary_w = cv2.morphologyEx(binary_w, cv2.MORPH_CLOSE, KERNEL).astype(np.uint8)
        print('m_close_binary_w done')
        

        # Invert
        new_b = mask_invert(m_close_binary_w, img_mask).astype(np.uint8)
        print('new_b done')
        

        # Morphological open
        m_open_new_b = cv2.morphologyEx(new_b, cv2.MORPH_OPEN,
                                        KERNEL).astype(np.uint8)
        print('m_open_new_b done')
        

        # Remove noise components inside coast
        new_clean = filter_components(m_open_new_b, transform).astype(np.uint8)
        print('new_clean done')
        
        # Revert back
        new_clean_invert = mask_invert(new_clean, img_mask).astype(np.uint8)
        print('new_clean_invert done')
       

        # Get a bounding box of the image extent
        gdf_inner = get_bounding_box(img_mask, transform)
        print('gdf_inner done')

        # Use bounding box to clip the reference coastline polygons, and dissolve them
        clipped_inner_dissolved = clip_ref_polygon_inner(gdf_inner)
        print('clipped_inner_dissolved done')

        # Get the polygons of the areas classified as land
        polygons = mask_to_polygons(new_clean_invert, clipped_inner_dissolved,
                                    two_d_transform)
        print('polygons done')

        # Get the polygons of land/inner features misclassified as ocean
        inner_polygons = mask_to_polygons(new_clean, clipped_inner_dissolved, two_d_transform, 10)
        print('inner polygons done')

        # Gather all resulting polygons
        polygons.extend(inner_polygons)
        print('extend polygons done')

        # Dissolve polygons to have a unified coastline polygon, and save to shapefile
        dissolved = polygons_to_shpfile(polygons, two_d_transform, dst_filename)
        print('dissolved to shapefile done')


        # Change detection using the change polygon method
        positive_change_polygons, positive_change_area = get_change(dissolved, clipped_inner_dissolved)
        negative_change_polygons, negative_change_area = get_change(clipped_inner_dissolved, dissolved)
        print('change done')
        # Calculate net change
        net_change_area = positive_change_area - negative_change_area
        
        # Save change polygons
        positive_change_polygons.to_file(dst_filename + '_pos_change_' + '.shp')
        negative_change_polygons.to_file(dst_filename + '_neg_change_' + '.shp')
        print('save change done')
        print('Positive change: '+ str(positive_change_area/1000000) + ' km^2')
        print('Negative change: ' + str(negative_change_area/1000000) + ' km^2')
        print('Net change: ' + str(net_change_area/1000000) + ' km^2')
        f.write('Positive change: '+ str(positive_change_area/1000000) + ' km^2' + '\n')
        f.write('Negative change: ' + str(negative_change_area/1000000) + ' km^2' + '\n')
        f.write('Net change: ' + str(net_change_area/1000000) + ' km^2' + '\n')
        f.write('----------------------------------------------------------------------' + '\n')
        
    execution_time = (time.time() - start_time)
    f.write("Finished processing images." + '\n')
    f.write("Execution time: " + str(execution_time) + '\n')
    f.write("Average time per image: " + str(execution_time/i))
    print("Finished processing images.")
    print("Execution time: " + str(execution_time))
    print("Average time per image: " + str(execution_time/i))
