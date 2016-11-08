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
import pylab
import tables
import scipy.ndimage.interpolation
import os.path as path

EXPECTED_SIZE = 224
EXPECTED_CHANNELS = 3
EXPECTED_DIM = (EXPECTED_CHANNELS, EXPECTED_SIZE, EXPECTED_SIZE)
EXPECTED_CLASS = 5
MAX_VALUE = 255


def square_slice_generator(data, size, slices_per_axis=5):
    if data.shape[1] <= size or data.shape[2] <= size:
        yield(resize(data, size))
    else:
        remaining_rows = data.shape[1] - size
        remaining_cols = data.shape[2] - size
        slide_delta_rows = remaining_rows / slices_per_axis
        slide_delta_cols = remaining_cols / slices_per_axis
        # print(data.shape)
        for i in range(slices_per_axis):
            row_start = i + i * slide_delta_rows
            row_end = row_start + size
            for j in range(slices_per_axis):
                col_start = j + j * slide_delta_cols
                col_end = col_start + size
                tmp = data[:, row_start:row_end, col_start:col_end]
                tmp = tmp * 1.0 / MAX_VALUE
                # print('({}:{}, {}:{})'.format(row_start, row_end, col_start, col_end))
                # plot_comparison((data, tmp))
                yield(tmp)


def resize(data, size):
    # calculate scale
    scale_row = 1.0 * size / data.shape[1]
    scale_col = 1.0 * size / data.shape[2]
    resized = scipy.ndimage.interpolation.zoom(data, (1, scale_row, scale_col), order=3, prefilter=True)
    resized = resized * 1.0 / MAX_VALUE
    # print(resized.shape)
    # plot_comparison((data, resized))
    return resized


def center_crop(data):
    # crop
    median_x = data.shape[0] / 2
    median_y = data.shape[1] / 2
    delta_x = data.shape[0] / 3
    delta_y = data.shape[1] / 3
    cropped = data[median_x - delta_x:median_x + delta_x, median_y - delta_y:median_y + delta_y]
    # render image
    # plot_comparison((data, cropped))
    return cropped


def plot_comparison(images):
    fig = pylab.figure()
    for i, m in enumerate(images):
        fig.add_subplot(len(images), 1, i + 1)
        pylab.imshow(m.transpose((1, 2, 0)))
    pylab.show()
    cv2.waitKey(0)


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
                im = cv2.imread(path_sub)
                transposed = im.transpose((2, 0, 1))
                # gather stat
                if transposed.shape not in stat:
                    stat[transposed.shape] = 0
                stat[transposed.shape] += 1
                for square in square_slice_generator(transposed, EXPECTED_SIZE):
                    # transpose to match VGG-16 input
                    data.append(np.array([square]))
                    # one-hot label
                    label = np.zeros(5)
                    label[i] = 1.0
                    labels.append(np.array([label]))
                # save test data
                data_test.append(np.array([resize(transposed, EXPECTED_SIZE)]))
                label = np.zeros(5)
                label[i] = 1.0
                labels_test.append(np.array([label]))
                # update progress bar
                bar.update(count)
                count += 1
        i += 1
bar.finish()

print('{} records saved'.format(data.nrows))

# write label index as json file
with open(index_file, 'w') as f:
    json.dump(label_indexes, f)

# close file
datafile.close()
testfile.close()
