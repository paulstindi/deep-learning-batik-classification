'''
Batik feature extractor
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python extractor.py <in:dataset.h5>
'''

import os
import tables
import sys
from keras.applications.vgg16 import VGG16


# params
BATCH_SIZE = 40

# const
FEATURES_FILE = 'features.h5'
FEATURES_DIM = (512, 7, 7)
EXPECTED_CLASS = 5

if __name__ == '__main__':
    # command line arguments
    dataset_file = sys.argv[1]
    features_file = sys.argv[2] if len(sys.argv) > 2 else FEATURES_FILE

    # loading dataset
    print('Loading preprocessed dataset: {}'.format(dataset_file))
    datafile = tables.open_file(dataset_file, mode='r')
    dataset = datafile.root
    print((dataset.data.nrows,) + dataset.data[0].shape)

    # feature extractor
    extractor = VGG16(weights='imagenet', include_top=False)

    print('Feature extraction')
    features_datafile = tables.open_file(features_file, mode='w')
    features_data = features_datafile.create_earray(features_datafile.root, 'data', tables.Float32Atom(shape=FEATURES_DIM), (0,), 'dream')
    features_labels = features_datafile.create_earray(features_datafile.root, 'labels', tables.UInt8Atom(shape=(EXPECTED_CLASS)), (0,), 'dream')
    i = 0
    while i < dataset.data.nrows:
        end = i + BATCH_SIZE
        data_chunk = dataset.data[i:end]
        label_chunk = dataset.labels[i:end]
        i = end
        features_data.append(extractor.predict(data_chunk, verbose=1))
        features_labels.append(label_chunk)

    assert features_datafile.root.data.nrows == dataset.data.nrows
    assert features_datafile.root.labels.nrows == dataset.labels.nrows

    # close feature file
    features_datafile.close()
