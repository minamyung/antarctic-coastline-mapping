import cv2
import numpy as np
import matplotlib.pyplot as plt

test_raster = cv2.imread('C:/Users/myung/Documents/CSC8099/Data/range_normalised/16bit_proper/S1B_EW_GRDM_1SDH_20210711T074538_20210711T074636_027743_034F9A_C432_Orb_TNR_Cal_RN_1_3031.tif', cv2.IMREAD_GRAYSCALE)

test_img = np.array([[1, 1, 1],
            [1, 0, 0],
            [2, 2, 1]]).astype(np.uint8)

def otsu_mask(img):
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
    return ret, thresh_img

no_zeros_retval, no_zeros_img = otsu_mask(test_raster)
with_zeros_retval, with_zeros_img = cv2.threshold(test_raster, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

print(no_zeros_retval)
print(with_zeros_retval)

plt.imshow(no_zeros_img)
plt.show()

plt.imshow(with_zeros_img)
plt.show()