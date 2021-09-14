# Testing the original code on a single image and show the output at different stages using pyplot
# Runs faster than original and removes image border

# MAKE SURE TO RUN WITH THE CORRECT CONDA ENV OTHERWISE YOU'LL LOSE IT COMPLETELY

import sys
from collections import defaultdict
import time
import glob
import cv2
from cv2 import ximgproc
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, ogr
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import MultiPolygon, Point, Polygon, box, shape
from shapely.affinity import affine_transform
import rasterio.features
import rasterio.mask
from affine import Affine

startTime = time.time()

# location of source images
filepath = 'C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/'

# location to save processed images
nfnb= 'C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/input/'
# nfnb = 'C:/Users/myung/Documents/CSC8099/Example_problems'

# images = ['S1B_EW_GRDM_1SDH_20210711T074538_20210711T074636_027743_034F9A_C432_Orb_TNR_Cal_RN_1_3031.tif']

images = []
for name in glob.glob(filepath + "*.tif"):
    trunc_name = str(name).split('\\')[-1]
    images.append(trunc_name) # Save the truncated (w/o path) image file names to make naming the result products easier


# Reference coastline for polygon masking
REF_PATH = 'C:/Users/myung/Documents/CSC8099/Data/add_coastline_high_res_polygon_v7_4/add_coastline_high_res_polygon_v7_4.shp'
REF_GS = gpd.GeoSeries.from_file(REF_PATH)
REF_GDF = gpd.GeoDataFrame.from_file(REF_PATH)


KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))

def read_img(filename):
    # read image with tiffile to handle larger rasters
    # img = tifi.imread(filename).astype(np.uint8)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    
    # read it as a GeoTiff to collect geodata
    geo_file = gdal.Open(filename)
    img_array = geo_file.GetRasterBand(1).ReadAsArray()

    # create a NO DATA mask
    bool_mask = (img_array != 0)
    img_mask = bool_mask.astype(np.uint8)
    

    return img, img_mask, geo_file

def b_filter(img):
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur

def diffuse(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    diffused = ximgproc.anisotropicDiffusion(img_rgb, 0.1, 10, 50)
    diffused_gray = cv2.cvtColor(diffused, cv2.COLOR_RGB2GRAY)
    return diffused_gray

def get_binary(img):
    # Takes a raster array (needs to be np.uint8) and returns its otsu threshold + the result of the thresholding
    # Calculated by disregarding black areas (border)

    # Remove the cells which have a value of 0 (border)
    bool_index = (img != 0)
    # Add them to an array
    temp_img = np.extract(bool_index, img)
    # Calculate Otsu threshold from the pixels which have a non-zero value
    thresh_val, temp = cv2.threshold(temp_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Apply thresholding to raster array with the threshold value calculated above
    ret, thresh_img = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    return thresh_img

def filter_components(img, geo_file):
    MIN_AREA = 1000000000 # Minimum area threshold in m^2 (1000km^2)

    # Get pixel sizes
    dx = geo_file.GetGeoTransform()[1]
    dy = -geo_file.GetGeoTransform()[5]

    # Scale the min component area using pixel size
    scaled_min_area = MIN_AREA / (dx * dy)

    # split into components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)

    # remove background
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # Make an array to hold filtered image
    filtered_img = np.zeros(output.shape)

    for i in range(0, nb_components):
        if sizes[i] > scaled_min_area:
            filtered_img[output == i + 1] = 1
    return filtered_img

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
    return img2, min_size


def get_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = map(np.squeeze, contours)
    polygons = map(Polygon, contours)
    #cv2.drawContours(img, contours, -1, (255, 0, 0), 250)
    return img

def mask_to_polygons(mask, clipped_inner, geo_file, epsilon=10, min_area=10.):
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    # approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
    #                    for cnt in contours]
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
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly.buffer(0))
    # approximating polygons might have created invalid ones, fix them
    # all_polygons = MultiPolygon(all_polygons)
    # if not all_polygons.is_valid:
    #     all_polygons = all_polygons.buffer(0)
    #     # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
    #     # need to keep it a Multi throughout
    #     if all_polygons.type == 'Polygon':
    #         all_polygons = MultiPolygon([all_polygons])

    # Create geoDataFrame
    transform = Affine.from_gdal(*geo_file.GetGeoTransform())
    # 2d transform method from https://gis.stackexchange.com/questions/380357/affine-tranformation-matrix-shapely-asks-6-coefficients-but-rasterio-delivers
    two_d_transform = [elem for tuple in transform.column_vectors for elem in tuple]
    
    polygons_names = []

    for poly in all_polygons:
        polygons_names.append('contour polygon')
    
    df_polygons = {'name': polygons_names, 'geometry': all_polygons}
    gdf_polygons = gpd.GeoDataFrame(df_polygons, crs='epsg:3031').affine_transform(two_d_transform)
    #print(gdf_polygons)
    #gdf_polygons.to_file('C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/all_ocean_new.shp')

    # retain only land polygons (remove misclassified ocean areas)
    inner_polygons = []
    for ref_polygon in clipped_inner['geometry']:
        #print('ref area' + str(ref_polygon.area))
        for i in range(len(gdf_polygons)):
            polygon = gdf_polygons[i]
            #print('polygon area' + str(polygon.area))
            intersect = ref_polygon.intersection(polygon)
            intersect_area = intersect.area
            #print('intersect area' + str(intersect_area))
            intersect_percentage = intersect_area / polygon.area
            #print('percentage' + str(intersect_percentage))
            if intersect_percentage > 0.9:
                inner_polygons.append(all_polygons[i])

    return inner_polygons

def mask_to_polygons_filter(mask, clipped_inner, geo_file, epsilon=10, min_area=10):
    # find misclassified inner features

    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    # approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
    #                    for cnt in contours]
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
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly.buffer(10))
    # approximating polygons might have created invalid ones, fix them
    # all_polygons = MultiPolygon(all_polygons)
    # if not all_polygons.is_valid:
    #     all_polygons = all_polygons.buffer(0)
    #     # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
    #     # need to keep it a Multi throughout
    #     if all_polygons.type == 'Polygon':
    #         all_polygons = MultiPolygon([all_polygons])

    # Create geoDataFrame
    transform = Affine.from_gdal(*geo_file.GetGeoTransform())
    # 2d transform method from https://gis.stackexchange.com/questions/380357/affine-tranformation-matrix-shapely-asks-6-coefficients-but-rasterio-delivers
    two_d_transform = [elem for tuple in transform.column_vectors for elem in tuple]
    
    polygons_names = []

    for poly in all_polygons:
        polygons_names.append('contour polygon')
    
    df_polygons = {'name': polygons_names, 'geometry': all_polygons}
    gdf_polygons = gpd.GeoDataFrame(df_polygons, crs='epsg:3031').affine_transform(two_d_transform)
    #print(gdf_polygons)
    #gdf_polygons.to_file('C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/all_ocean_new.shp')

    # retain only misclassified polygons that should be land
    inner_polygons = []
    for ref_polygon in clipped_inner['geometry']:
        #print('ref area' + str(ref_polygon.area))
        for i in range(len(gdf_polygons)):
            polygon = gdf_polygons[i]
            #print('polygon area' + str(polygon.area))
            intersect = ref_polygon.intersection(polygon)
            intersect_area = intersect.area
            #print('intersect area' + str(intersect_area))
            intersect_percentage = intersect_area / polygon.area
            #print('percentage' + str(intersect_percentage))
            if intersect_percentage > 0.9:
                inner_polygons.append(all_polygons[i])
    return inner_polygons

def polygons_to_shpfile(polygons, geo_file):
    transform = Affine.from_gdal(*geo_file.GetGeoTransform())
    # 2d transform method from https://gis.stackexchange.com/questions/380357/affine-tranformation-matrix-shapely-asks-6-coefficients-but-rasterio-delivers
    two_d_transform = [elem for tuple in transform.column_vectors for elem in tuple]
    
    polygons_transformed = []
    polygons_names = []
    
    for poly in polygons:
        polygons_transformed.append(affine_transform(poly, two_d_transform))
        polygons_names.append('contour polygon')
    
    df_polygons = {'name': polygons_names, 'geometry': polygons_transformed}
    gdf_polygons = gpd.GeoDataFrame(df_polygons, crs='epsg:3031')
    gdf_dissolved = gdf_polygons.dissolve()


    gdf_dissolved.to_file('C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/percentage_pls_work_7_dissolved.shp')

def extract_polygons(img, geo_file, img_mask):
    # Removes internal features of detected coastline using the reference polygon coastline

    # Initialise polygons list
    transform = Affine.from_gdal(*geo_file.GetGeoTransform())
    polygons = []
    polygons_names = []

    # Go through all the shapes extracted from the img array 
    for vec in rasterio.features.shapes(img, mask=img_mask, transform=transform): 
        polygons.append(shape(vec[0]).buffer(0)) # Save as a shape
        polygons_names.append('extracted polygons')
    
    df_polygons = {'name': polygons_names, 'geometry': polygons}
    gdf_polygons = gpd.GeoDataFrame(df_polygons, crs='epsg:3031')
    #gdf_polygons.to_file('C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/extracted_polygons_new.shp')
    #clip_polygons = gpd.clip(gdf_polygons, gdf_inner)
    #clip_polygons.to_file('C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/clipped_polygons_new.shp')
    return gdf_polygons

def get_bounding_box(img_mask, geo_file):
    # Returns a polygon of the size of the image bounds
    # Get Affine transform of original image
    transform1 = Affine.from_gdal(*geo_file.GetGeoTransform())
    # Initialise mask_polygon
    mask_polygon = None
    # Go through all the shapes extracted from the img_mask array 
    for vec in rasterio.features.shapes(img_mask, transform=transform1): 
        if(vec[1] == 1): # The inner box of the img_mask is where the value is 1
            mask_polygon = shape(vec[0]) # Save as a shape
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

def polygon_filtering(extracted_polygons, clipped_inner, geo_file):
    # # Buffer the reference coastline polygon(s)
    # polygons = []
    # polygons_names = []
    # for ref_polygon in clipped_inner['geometry']:
    #     polygons.append(ref_polygon.buffer(250))
    #     polygons_names.append('buffered ref polygons')
    # df_buffered_polygons = {'name': buffered_polygons_names, 'geometry': buffered_polygons}
    # gdf_buffered_polygons = gpd.GeoDataFrame(df_buffered_polygons, crs='epsg:3031')
    # gdf_buffered_polygons.to_file('C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/percentage/90.shp')
    
    # For each of the extracted polygons, check whether it is entirely contained within the ref coastline polygons. If so, it can be disregarded as an internal feature to get a definitive coastline.
    # extracted_polygons is a layer
    filtered_polygons = []
    filtered_polygons_names = []
    for ref_polygon in clipped_inner['geometry']:
        for polygon in extracted_polygons:
            coverage_area = ref_polygon.intersection(polygon).area
            coverage_percentage = coverage_area / polygon.area
            
            if coverage_percentage < 0.9:
                filtered_polygons.append(polygon)
                filtered_polygons_names.append('filtered polygons')
            # Clipped_inner might have more than one polygon so need to iterate over polygons here
            # if not ref_polygon.contains(polygon):
            #     filtered_polygons.append(polygon)
            #     filtered_polygons_names.append('filtered polygons')
    # df_filtered_polygons = {'name': filtered_polygons_names, 'geometry': filtered_polygons}
    # gdf_filtered_polygons = gpd.GeoDataFrame(df_filtered_polygons, crs='epsg:3031')
    # gdf_filtered_polygons.to_file('C:/Users/myung/Documents/CSC8099/Data/polygon_filtering/filtered.shp')
    return filtered_polygons

def remove_mask(img, img_mask):
    img2 = cv2.bitwise_xor(img,img_mask)
    return img2

def remove_border(img, boundary):
    img2 = cv2.bitwise_and(img, img, mask=boundary)
    return img2

def save_gtiff(img, nfn):
    print(nfn)
    driver_tiff = gdal.GetDriverByName("GTiff")

    # create GeoTiff
    nds = driver_tiff.Create(nfn, xsize=geo_file.RasterXSize, ysize=geo_file.RasterYSize, bands=1,
                                eType=gdal.GDT_UInt16)
    nds.SetGeoTransform(geo_file.GetGeoTransform())
    nds.SetProjection(geo_file.GetProjection())

    # copy output to the created file
    bandn = nds.GetRasterBand(1).ReadAsArray()
    bandn = img
    nds.GetRasterBand(1).WriteArray(bandn)

    # set no data to remove background
    nds.GetRasterBand(1).SetNoDataValue(0)
    nds = None

i = 0 # Count the number of images processed

for image_name in images:
    i += 1
    total = len(images)

    filename = filepath + image_name

    # set the new file name (same as source image, but different folder)
    nfn = nfnb + image_name 

    print(str(i) + "/" + str(total))
    print('Processing ' + image_name)

    (img, img_mask, geo_file) = read_img(filename)

    #cv2.imwrite('C:/Users/myung/Documents/CSC8099/Example_problems/polygon_filtering/diffuse/0_img.jpg', img)

    blur = b_filter(img).astype(np.uint8)

    #cv2.imwrite('C:/Users/myung/Documents/CSC8099/Example_problems/polygon_filtering/diffuse/1_blur.jpg', blur)
    # plt.imshow(blur)
    # plt.show()

    # cv2.imwrite('C:/Users/myung/Documents/CSC8099/Example_problems/polygon_filtering/diffuse/2_diffused_50iter.jpg', diffused)
    # save_gtiff(diffused, nfn)

    binary = get_binary(blur).astype(np.uint8)
    # cv2.imwrite('C:/Users/myung/Documents/CSC8099/Example_problems/polygon_filtering/diffuse/3_binary_50iter.jpg', binary)
    # plt.imshow(binary)
    # plt.show()


    # Remove components in ocean
    binary_w = filter_components(binary, geo_file).astype(np.uint8)
    m_close_binary_w = cv2.morphologyEx(binary_w, cv2.MORPH_CLOSE, KERNEL)
    # plt.imshow(binary_w)
    # plt.show()

    new_b = remove_mask(m_close_binary_w, img_mask).astype(np.uint8)
    # plt.imshow(new_b)
    # plt.show()

    m_open_new_b = cv2.morphologyEx(new_b, cv2.MORPH_OPEN, KERNEL).astype(np.uint8)
    # plt.imshow(m_open_new_b)
    # plt.show()
    
    # Remove small internal features
    new_clean = filter_components(m_open_new_b, geo_file).astype(np.uint8)
    # plt.imshow(new_clean)
    # plt.show()
    
    # Revert back
    new_clean_invert = remove_mask(new_clean, img_mask).astype(np.uint8)
    # plt.imshow(new_clean_invert)
    # plt.show()

 
    gdf_inner = get_bounding_box(img_mask, geo_file)
    clipped_inner_dissolved = clip_ref_polygon_inner(gdf_inner)

    # multipolygon = mask_to_polygons(img)
    polygons = mask_to_polygons(new_clean_invert, clipped_inner_dissolved, geo_file)

    inner_polygons = mask_to_polygons_filter(new_clean, clipped_inner_dissolved, geo_file)

    polygons.extend(inner_polygons)
    
    polygons_to_shpfile(polygons, geo_file)
    # Find contours and display
    # contoured_img = get_contours(new_clean_invert)
    # plt.imshow(contoured_img)
    # plt.show()
    

    sys.exit()


    # Remove internal components
    gdf_inner = get_bounding_box(img_mask, geo_file)
    clipped_inner_dissolved = clip_ref_polygon_inner(gdf_inner)

    extracted_polygons = extract_polygons(m_open_new_b, geo_file, img_mask)
  

    new_clean = polygon_filtering(extracted_polygons, clipped_inner_dissolved, geo_file)
    # new_clean = filter_components(new_b, geo_file).astype(np.uint8)
    # plt.imshow(new_clean)
    # plt.show()
    sys.exit()
    
    temp = remove_mask(new_clean, img_mask).astype(np.uint8)
    plt.imshow(temp)
    plt.show()
    
    binary_clean = (~temp.astype(bool)).astype(np.uint8)


    border_boundary = img_mask - ndimage.morphology.binary_dilation(img_mask)

    # Dilate the border of the image so we can exclude this from the final boundary line
    dilation_kernel = np.ones((50,50),np.uint8)
    dilated_border = cv2.dilate(border_boundary,dilation_kernel,iterations = 1)
    border_mask = cv2.bitwise_not(dilated_border)

    boundary = binary_clean - ndimage.morphology.binary_dilation(binary_clean)
    noborder_boundary = remove_border(boundary, border_mask).astype(np.uint8)
    kernel = np.ones((3, 3))
    final = cv2.morphologyEx(noborder_boundary, cv2.MORPH_CLOSE, kernel)
    plt.imshow(final)
    plt.show()


    driver_tiff = gdal.GetDriverByName("GTiff")

    # set the new file name (same as source image, but different folder)
    nfn = nfnb + image_name 
    # print(nfn)

    # create GeoTiff
    nds = driver_tiff.Create(nfn, xsize=geo_file.RasterXSize, ysize=geo_file.RasterYSize, bands=1,
                                eType=gdal.GDT_UInt16)
    nds.SetGeoTransform(geo_file.GetGeoTransform())
    nds.SetProjection(geo_file.GetProjection())

    # copy output to the created file
    bandn = nds.GetRasterBand(1).ReadAsArray()
    bandn = final
    nds.GetRasterBand(1).WriteArray(bandn)

    # set no data to remove background
    nds.GetRasterBand(1).SetNoDataValue(0)
    nds = None

executionTime = (time.time() - startTime)
print("Finished processing images.")
print("Execution time: " + str(executionTime))