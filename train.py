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
import json
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# training parameters
BATCH_SIZE = 50
NB_EPOCH = 100
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
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    class_file = sys.argv[3]

    print('BATCH_SIZE: {}'.format(BATCH_SIZE))
    print('NB_EPOCH: {}'.format(NB_EPOCH))

    # loading dataset
    print('Loading train dataset: {}'.format(train_file))
    train_datafile = tables.open_file(train_file, mode='r')
    train_dataset = train_datafile.root
    print('Train data: {}'.format((train_dataset.data.nrows,) + train_dataset.data[0].shape))

    print('Loading test dataset: {}'.format(test_file))
    test_datafile = tables.open_file(test_file, mode='r')
    test_dataset = test_datafile.root
    print('Test data: {}'.format((test_dataset.data.nrows,) + test_dataset.data[0].shape))

    # setup model
    print('Preparing model')
    model = Sequential()
    model.add(Flatten(input_shape=FEATURES_DIM))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.6))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.6))
    model.add(Dense(EXPECTED_CLASS, activation='softmax', init='uniform'))

    # compile the model (should be done *after* setting layers to non-trainable)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # training model
    num_rows = train_dataset.data.nrows
    if num_rows > DATASET_BATCH_SIZE:
        # batch training
        print('DATASET_BATCH_SIZE: {}'.format(DATASET_BATCH_SIZE))
        model.fit_generator(
            dataset_generator(train_dataset, BATCH_SIZE),
            samples_per_epoch=num_rows,
            nb_epoch=NB_EPOCH,
            validation_data=dataset_generator(test_dataset, DATASET_BATCH_SIZE)
        )
    else:
        # one-go training
        print('BATCH_SIZE: {}'.format(BATCH_SIZE))
        X_train = train_dataset.data[:]
        X_test = test_dataset.data[:]
        Y_train = train_dataset.labels[:]
        Y_test = test_dataset.labels[:]

        # # this will do preprocessing and realtime data augmentation
        # datagen = ImageDataGenerator(
        #     featurewise_center=False,  # set input mean to 0 over the dataset
        #     samplewise_center=False,  # set each sample mean to 0
        #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #     samplewise_std_normalization=False,  # divide each input by its std
        #     zca_whitening=False,  # apply ZCA whitening
        #     rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
        #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        #     horizontal_flip=True,  # randomly flip images
        #     vertical_flip=False)  # randomly flip images
        # datagen.fit(X_train)
        # model.fit_generator(
        #     datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
        #     samples_per_epoch=8000,
        #     nb_epoch=NB_EPOCH,
        #     validation_data=(X_test, Y_test))

        model.fit(X_train, Y_train,
                  batch_size=BATCH_SIZE,
                  nb_epoch=NB_EPOCH,
                  validation_data=(X_test, Y_test),
                  shuffle=True)

        print('Predicting')
        # set max class to 1 and rest to 0
        Y_pred = model.predict_on_batch(X_test)
        predictions = Y_pred.argmax(1)
        truths = Y_test.argmax(1)
        acc = accuracy_score(truths, predictions)
        print('Accuracy: {}'.format(acc))
        print('Confusion matrix: ')
        cm = confusion_matrix(truths, predictions)
        print(cm)

    with open(class_file, 'r') as f:
        classes = json.load(f)
        print(['{}:{}'.format(i, classes[i]) for i in sorted(classes)])

    # saving model
    # print('Saving model')
    # model.save(MODEL_NAME)

    # close dataset
    train_datafile.close()
    test_datafile.close()
