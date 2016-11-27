'''
Train batik classifier
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py <in:dataset.h5> <in:dataset.test.h5>
'''
from __future__ import print_function
import numpy as np
import sys
import cv2
import json
import tables
import math
import random
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from datetime import datetime

# training parameters
BATCH_SIZE = 40
NB_EPOCH = 5
DATASET_BATCH_SIZE = 1000

# const
EXPECTED_SIZE = 224
EXPECTED_CHANNELS = 3
EXPECTED_DIM = (EXPECTED_CHANNELS, EXPECTED_SIZE, EXPECTED_SIZE)
EXPECTED_CLASS = 5
MODEL_NAME = 'model.h5'


def dataset_generator(dataset, batch_size):
    while True:
        i = 0
        while i < dataset.data.nrows:
            end = i + batch_size
            X = dataset.data[i:end]
            Y = dataset.labels[i:end]
            i = end
            yield(X, Y)


# command line arguments
dataset_file = sys.argv[1]
test_file = sys.argv[2]

print('BATCH_SIZE: {}'.format(BATCH_SIZE))
print('NB_EPOCH: {}'.format(NB_EPOCH))

# loading dataset
print('Loading test & train dataset: {}'.format(dataset_file))
datafile = tables.open_file(dataset_file, mode='r')
dataset = datafile.root
print('Train data: {}'.format((dataset.data.nrows,) + dataset.data[0].shape))
test_tables = tables.open_file(test_file, mode='r')
test = test_tables.root
X_test = test.data[:]
Y_test = test.labels[:]
print('Test data: {}'.format(X_test.shape))

# setup model
print('Preparing model')
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=EXPECTED_DIM))
x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(EXPECTED_CLASS, activation='softmax', init='uniform')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# freeze base_model layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# training model
num_rows = dataset.data.nrows
if num_rows > DATASET_BATCH_SIZE:
    # batch training
    print('DATASET_BATCH_SIZE: {}'.format(DATASET_BATCH_SIZE))
    model.fit_generator(
        dataset_generator(dataset, BATCH_SIZE),
        samples_per_epoch=num_rows,
        nb_epoch=NB_EPOCH,
        validation_data=(X_test, Y_test)
    )
else:
    # one-go training
    X_train = dataset.data[:]
    Y_train = dataset.labels[:]
    model.fit(X_train, Y_train,
              batch_size=BATCH_SIZE,
              nb_epoch=NB_EPOCH,
              validation_data=(X_test, Y_test),
              shuffle=True)

# saving model
print('Saving model')
model.save(MODEL_NAME)

# close dataset
datafile.close()
