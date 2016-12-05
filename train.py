'''
Train batik classifier
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py <in:features.h5>
'''
from __future__ import print_function
import numpy as np
import sys
import cv2
import tables
import math
import random
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

# training parameters
BATCH_SIZE = 40
NB_EPOCH = 5
DATASET_BATCH_SIZE = 1000

# const
FEATURES_DIM = (512, 7, 7)
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


if __name__ == '__main__':
    # command line arguments
    dataset_file = sys.argv[1]

    print('BATCH_SIZE: {}'.format(BATCH_SIZE))
    print('NB_EPOCH: {}'.format(NB_EPOCH))

    # loading dataset
    print('Loading train dataset: {}'.format(dataset_file))
    datafile = tables.open_file(dataset_file, mode='r')
    dataset = datafile.root
    print('Train data: {}'.format((dataset.data.nrows,) + dataset.data[0].shape))

    # setup model
    print('Preparing model')
    model = Sequential()
    model.add(Flatten(input_shape=FEATURES_DIM))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(EXPECTED_CLASS, activation='softmax', init='uniform'))

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
            nb_epoch=NB_EPOCH
        )
    else:
        # one-go training
        print('BATCH_SIZE: {}'.format(BATCH_SIZE))
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.data[:], dataset.labels[:], test_size=0.1, random_state=42)
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
