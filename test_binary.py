# Testing the original code on a single image and show the output at different stages using pyplot
# Runs faster than original and removes image border

# MAKE SURE TO RUN WITH THE CORRECT CONDA ENV OTHERWISE YOU'LL LOSE IT COMPLETELY


import time
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from scipy import ndimage
import sys

startTime = time.time()

# location of source images
filepath = 'C:/Users/myung/Documents/CSC8099/Data/range_normalised/'

# location to save processed images
nfnb= 'C:/Users/myung/Documents/CSC8099/Data/Input_rn/'


# images = ['S1B_IW_GRDH_1SSH_20210711T043551_B06E_S_1.tif']
images = []
for name in glob.glob(filepath + "*.tif"):
    trunc_name = str(name).split('\\')[-1]
    images.append(trunc_name) # Save the truncated (w/o path) image file names to make naming the result products easier

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

    (image, mask, geo_file) = read_img(filename)
    blur = b_filter(image).astype(np.uint8)
    # plt.imshow(blur)
    # plt.show()
    binary = get_binary(blur).astype(np.uint8)
 
    binary_w = delete_b(binary, 0.05).astype(np.uint8)

    new_b = remove_mask(binary_w, mask).astype(np.uint8)
    # plt.imshow(new_b)
    # plt.show()
    new_clean = delete_b(new_b, 0.05).astype(np.uint8)
    # plt.imshow(new_clean)
    # plt.show()
    temp = remove_mask(new_clean, mask).astype(np.uint8)
    # plt.imshow(temp)
    # plt.show()
    binary_clean = (~temp.astype(bool)).astype(np.uint8)


    border_boundary = mask - ndimage.morphology.binary_dilation(mask)

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
    nfn = nfnb + '0.05_areas' + image_name 


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