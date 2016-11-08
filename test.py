'''
Test batik classification model
Usage: python test.py <in:model file> <in:test dataset>

'''

import tables
import sys
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.utils import np_utils, generic_utils


model_file = sys.argv[1]
test_file = sys.argv[2]

print('Loading model')
model = load_model(model_file)
# sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

print('Loading test dataset')
test_tables = tables.open_file(test_file, mode='r')
test = test_tables.root
X = test.data[:]
Y = test.labels[:]

# prediction = model.predict(X[0, :, :, :])
# print(prediction)
# print(Y[0:5])

# print(X[:10].shape)
# print(Y[:10].shape)
print('Evaluating')
score = model.evaluate(X, Y)
print "%s: %.2f%%" % (model.metrics_names[1], score[1] * 100)

test_tables.close()
print('Done.')
