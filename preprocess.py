'''
Vectorize batik dataset
Usage: python preprocess.py <batik images directory>

'''
import os
import sys
import cv2
import numpy as np
import json
import progressbar

# get absolute path from arg
mypath = sys.argv[1]
dataset_file = sys.argv[2]
index_file = sys.argv[3]

# iterate dir content
label_indexes = {}
labels = []
data = []
bar = progressbar.ProgressBar(max_value=1000)
count = 1
i = 0
for f in os.listdir(mypath):
    path = os.path.join(mypath, f)
    if os.path.isdir(path) and f != 'Mix motif':
        label_indexes[i] = f
        for f_sub in os.listdir(path):
            path_sub = os.path.join(path, f_sub)
            if os.path.isfile(path_sub):
                im = cv2.resize(cv2.imread(path_sub), (224, 224)).astype(np.float32)
                # TODO calculate properly
                im[:, :, 0] -= 103.939
                im[:, :, 1] -= 116.779
                im[:, :, 2] -= 123.68
                im = im.transpose((2, 0, 1))
                data.append(im)
                labels.append(i)
                bar.update(count)
                count = count + 1
        i = i + 1
bar.finish()

# convert to numpy
x = np.array(data)
y = np.array(labels)

# write dataset to file
np.savez(dataset_file, x=x, y=y)

# write label index as json file
with open(index_file, 'w') as f:
    json.dump(label_indexes, f)

print(label_indexes)
print(x.shape)
print(y.shape)
