'''
Train batik classifier
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py <in:dataset.h5> <in:dataset.test.h5> <dataset.index.json>

'''
from __future__ import print_function
import numpy as np
import sys
import cv2
import json
import tables
import math
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from datetime import datetime

# training parameters
BATCH_SIZE = 50
NB_EPOCH = 1

# dataset
DATASET_BATCH_SIZE = 500


def VGG_16(weights_path=None, input_shape=(3, 224, 224), nb_class=5):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    # remove last layer
    model.layers.pop()
    model.add(Dense(nb_class, activation='softmax'))
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# command line arguments
dataset_file = sys.argv[1]
test_file = sys.argv[2]
dataset_index_file = sys.argv[3]

# read index
with open(dataset_index_file, 'r') as f:
    dataset_index = json.load(f)

# loading dataset
print('Loading dataset')
# npzfile = np.load(dataset_file)
# X = npzfile['x']
# Y = npzfile['y']
datafile = tables.open_file(dataset_file, mode='r')
dataset = datafile.root

# setup model
print('Preparing model')
model = VGG_16('../vgg16_weights.h5', dataset.data[0].shape, len(dataset_index))
# model = VGG_16(None, dataset.data[0].shape, len(dataset_index))


# training model
num_rows = dataset.data.nrows
num_iterate = num_rows / DATASET_BATCH_SIZE
print('Training model using {} data in batch of {}'.format(num_rows, DATASET_BATCH_SIZE))
for e in range(NB_EPOCH):
    print('Epoch {}/{}'.format(e + 1, NB_EPOCH))
    for i in range(num_iterate):
        print('Data batch {}/{}'.format(i + 1, num_iterate))
        begin = i + i * DATASET_BATCH_SIZE
        end = begin + DATASET_BATCH_SIZE
        X_train, X_test, Y_train, Y_test = train_test_split(
            dataset.data[begin:end], dataset.labels[begin:end],
            test_size=0.10
        )
        model.fit(X_train, Y_train,
                  batch_size=BATCH_SIZE,
                  nb_epoch=1,
                  validation_data=(X_test, Y_test),
                  shuffle=True)

# saving model
print('Saving model')
model.save('model_{}.h5'.format(datetime.now().strftime('%Y%m%d%H%M%S')))

print('Loading test dataset')
test_tables = tables.open_file(test_file, mode='r')
test = test_tables.root
X = test.data[:]
Y = test.labels[:]

print('Evaluating')
score = model.evaluate(X, Y)
print('{}: {}%'.format(model.metrics_names[1], score[1] * 100))

# close dataset
datafile.close()
