import cv2
import numpy as np

test_img = np.array([[1, 1, 1],
            [1, 0, 0],
            [0, 0, 1]])

def otsu_mask(img):
    # max_val = 256 # Maximum value for 8bit unsigned integer
    bool_index = (img != 0)
    temp_img = np.extract(bool_index, img)
    thresh_val, temp = cv2.threshold(temp_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, thresh_img = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    return thresh_img

otsu_mask(test_img)