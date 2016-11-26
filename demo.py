import sys
import cv2
import numpy as np

# config
EXPECTED_MAX = 100.0
EXPECTED_MIN = -1 * EXPECTED_MAX
FILTER_THRESHOLD = -90.0

# const
MAX_VALUE = 255
MEDIAN_VALUE = MAX_VALUE / 2.0
EXPECTED_SIZE = 224


def square_slice_generator(data, size, slices_per_axis=3):
    if data.shape[0] <= size or data.shape[1] <= size:
        yield(resize(EXPECTED_SIZE))
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


def resize(data, expected_size):
    resized = cv2.resize(data, (expected_size, expected_size))
    return resized


def normalize_and_filter(data, max_value=MAX_VALUE, median=MEDIAN_VALUE, expected_max=EXPECTED_MAX, expected_min=EXPECTED_MIN, threshold=FILTER_THRESHOLD):
    # data = cv2.bilateralFilter(data, 10, 50, 50)
    # data = cv2.medianBlur(data, 3)
    # data = cv2.GaussianBlur(data, (5, 5), 0)
    data = (data - median) / median * expected_max
    # data[data < threshold] = expected_min
    return data


img_file = sys.argv[1] if len(sys.argv) > 1 else 'batik-parang.jpg'
original = cv2.imread(img_file)
cv2.imshow('Original', original)
# print(original)
# print(original.shape)

normalized = normalize_and_filter(original)
cv2.imshow('Normalized', normalized)
print(normalized)
count = 1
for square in square_slice_generator(normalized, EXPECTED_SIZE):
    # cv2.imshow('Slice {}'.format(count), square)
    print(square.shape)
    count += 1

# # convert grayscale & transpose
# gray = cv2.imread(sys.argv[1], 0)
# gray = normalize_and_filter(gray)
# i = 1
# for square in square_slice_generator(gray, 224):
#     # just display few images as sample
#     if i in [1, 9]:
#         cv2.imshow('Slice {}'.format(i), square)
#     i += 1
cv2.waitKey(0)
