'''
Vectorize batik dataset
Usage: python preprocess.py <in:batik images directory> <out:dataset> <out.dataset.test> <out:dataset.index>

'''
import os
import sys
import cv2
import numpy as np
import json
import progressbar
import tables
from scipy import ndimage

# config
FILTER_THRESHOLD = 100

# global vars
EXPECTED_SIZE = 224
EXPECTED_CHANNELS = 3
EXPECTED_DIM = (EXPECTED_CHANNELS, EXPECTED_SIZE, EXPECTED_SIZE)
EXPECTED_CLASS = 5
MAX_VALUE = 255


def square_slice_generator(data, size, slices_per_axis=5):
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
    data[data < threshold] = 0
    # histogram equalization
    # data = cv2.equalizeHist(data)
    data = data * 1.0 / max_value
    # data = ndimage.gaussian_filter(data, 2)
    # data = ndimage.median_filter(data, 4)
    return data


def append_data_and_label(m, c, dataset, labels):
    # append as 3 channels
    dataset.append(np.array([[m, m, m]]))
    # one-hot encoding
    label = np.zeros(5)
    label[c] = 1.0
    labels.append(np.array([label]))


if __name__ == '__main__':
    # get absolute path from arg
    mypath = sys.argv[1]
    dataset_file = sys.argv[2]
    test_file = sys.argv[3]
    index_file = sys.argv[4]

    # iterate dir content
    stat = {}
    label_indexes = {}
    bar = progressbar.ProgressBar(maxval=700).start()
    count = 1
    i = 0

    # pytables file
    datafile = tables.open_file(dataset_file, mode='w')
    data = datafile.create_earray(datafile.root, 'data', tables.Float32Atom(shape=EXPECTED_DIM), (0,), 'batik')
    labels = datafile.create_earray(datafile.root, 'labels', tables.UInt8Atom(shape=(EXPECTED_CLASS)), (0,), 'batik')

    testfile = tables.open_file(test_file, mode='w')
    data_test = testfile.create_earray(testfile.root, 'data', tables.Float32Atom(shape=EXPECTED_DIM), (0,), 'batik')
    labels_test = testfile.create_earray(testfile.root, 'labels', tables.UInt8Atom(shape=(EXPECTED_CLASS)), (0,), 'batik')

    # iterate subfolders
    for f in os.listdir(mypath):
        path = os.path.join(mypath, f)
        # exclude Mix motif
        if os.path.isdir(path) and f != 'Mix motif':
            label_indexes[i] = f
            for f_sub in os.listdir(path):
                path_sub = os.path.join(path, f_sub)
                if os.path.isfile(path_sub):
                    # read as gray image
                    gray = cv2.imread(path_sub, 0)
                    gray = normalize_and_filter(gray)
                    # gather stat
                    stat[gray.shape] = stat[gray.shape] + 1 if gray.shape in stat else 1
                    for square in square_slice_generator(gray, EXPECTED_SIZE):
                        # save train data
                        append_data_and_label(square, i, data, labels)
                    r = resize(gray, EXPECTED_SIZE)
                    append_data_and_label(r, i, data, labels)
                    # save test data
                    append_data_and_label(r, i, data_test, labels_test)
                    # update progress bar
                    bar.update(count)
                    count += 1
            i += 1
    bar.finish()

    print('{} records saved'.format(data.nrows))

    # write label index as json file
    with open(index_file, 'w') as f:
        json.dump(label_indexes, f)

    print((data.nrows,) + data[0].shape)
    print((labels.nrows,) + labels[0].shape)
    print(label_indexes)
    # print(stat)
    assert data[0].shape == EXPECTED_DIM
    assert labels[0].shape == (EXPECTED_CLASS,)

    # close file
    datafile.close()
    testfile.close()
