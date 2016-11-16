import sys
import cv2
import numpy as np
from scipy import ndimage

MAX_VALUE = 255
FILTER_THRESHOLD = 100


def square_slice_generator(data, size, slices_per_axis=3):
    if data.shape[0] <= size or data.shape[1] <= size:
        yield(resize(data, size))
    else:
        remaining_rows = data.shape[0] - size
        remaining_cols = data.shape[1] - size
        slide_delta_rows = remaining_rows / slices_per_axis
        slide_delta_cols = remaining_cols / slices_per_axis
        for i in range(slices_per_axis):
            row_start = i + i * slide_delta_rows
            row_end = row_start + size
            for j in range(slices_per_axis):
                col_start = j + j * slide_delta_cols
                col_end = col_start + size
                tmp = data[row_start:row_end, col_start:col_end]
                yield(tmp)


def resize(data, size):
    scale_row = 1.0 * size / data.shape[0]
    scale_col = 1.0 * size / data.shape[1]
    resized = ndimage.interpolation.zoom(data, (scale_row, scale_col), order=3, prefilter=True)
    return resized


def normalize_and_filter(data, max_value=MAX_VALUE, threshold=FILTER_THRESHOLD):
    # invertion
    # data = (255 - data)
    # threshold
    data[data < threshold] = 0
    # histogram equalization
    # data = cv2.equalizeHist(data)
    # data = ndimage.median_filter(data, 4)
    # data = ndimage.gaussian_filter(data, 2)
    data = data * 1.0 / max_value
    return data

original = cv2.imread(sys.argv[1])
cv2.imshow("Original", original)
# convert grayscale & transpose
gray = cv2.imread(sys.argv[1], 0)
gray = normalize_and_filter(gray)
i = 1
for square in square_slice_generator(gray, 224):
    # just display few images as sample
    if i in [1, 9]:
        cv2.imshow("Slice {}".format(i), square)
    i += 1
cv2.waitKey(0)
