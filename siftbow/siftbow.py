# -*- coding: utf-8 -*-
"""
Author: yohanes.gultom@gmail.com
Original reference: https://github.com/briansrls/SIFTBOW
"""

import cv2
import numpy as np
import os
import sys
import progressbar
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

# config
dictionarySize = 5
bow_sift_dictionary = 'bow_sift_dictionary.pkl'
bow_sift_features = 'bow_sift_features.pkl'
bow_sift_features_labels = 'bow_sift_features_labels.pkl'
svm_model = 'svm_model.xml'


if __name__ == '__main__':
    train_dir_path = sys.argv[1]
    test_dir_path = sys.argv[2]

    sift = cv2.SIFT()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    sift2 = cv2.DescriptorExtractor_create("SIFT")
    bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
    svm = cv2.SVM()

    if os.path.isfile(bow_sift_dictionary) and os.path.isfile(svm_model):
        dictionary = pickle.load(open(bow_sift_dictionary, "rb"))
        bowDiction.setVocabulary(dictionary)
        svm.load(svm_model)
        print('Loaded from {}'.format(bow_sift_dictionary))
        print('Loaded from {}'.format(svm_model))
    else:
        train_desc = []
        train_labels = []

        if os.path.isfile(bow_sift_dictionary) and os.path.isfile(bow_sift_features) and os.path.isfile(bow_sift_features_labels):
            dictionary = pickle.load(open(bow_sift_dictionary, "rb"))
            train_desc = pickle.load(open(bow_sift_features, "rb"))
            train_labels = pickle.load(open(bow_sift_features_labels, "rb"))
            print('Loaded from {}'.format(bow_sift_dictionary))
            print('Loaded from {}'.format(bow_sift_features))
            print('Loaded from {}'.format(bow_sift_features_labels))
            bowDiction.setVocabulary(dictionary)
        else:
            # list of our class names
            training_names = os.listdir(train_dir_path)
            training_paths = []

            # get full list of all training images
            print('Collecting training data..')
            for i in range(len(training_names)):
                p = training_names[i]
                subdir = os.path.join(train_dir_path, p)
                print(subdir)
                training_paths1 = os.listdir(subdir)
                for j in training_paths1:
                    training_paths.append(os.path.join(subdir, j))
                    train_labels.append(i)

            BOW = cv2.BOWKMeansTrainer(dictionarySize)
            print('Computing SIFT descriptors..')
            num_files = len(training_paths)
            bar = progressbar.ProgressBar(maxval=num_files).start()
            for i in range(num_files):
                p = training_paths[i]
                image = cv2.imread(p)
                gray = cv2.cvtColor(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                kp, dsc = sift.detectAndCompute(gray, None)
                BOW.add(dsc)
                bar.update(i)
            bar.finish()

            # creating dictionary/vocabularies
            print('Creating BoW vocabs using K-Means clustering with k={}..'.format(dictionarySize))
            dictionary = BOW.cluster()
            print "bow dictionary", np.shape(dictionary)
            pickle.dump(dictionary, open(bow_sift_dictionary, "wb"))
            bowDiction.setVocabulary(dictionary)

            print('Computing features using Bag-of-Words dictionary..')
            bar = progressbar.ProgressBar(maxval=num_files).start()
            for i in range(num_files):
                p = training_paths[i]
                im = cv2.imread(p, 1)
                gray = cv2.cvtColor(im, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                feature = bowDiction.compute(gray, sift.detect(gray))
                train_desc.extend(feature)
                bar.update(i)
            bar.finish()
            # save to file
            pickle.dump(train_desc, open(bow_sift_features, "wb"))
            pickle.dump(train_labels, open(bow_sift_features_labels, "wb"))

        print('Train SVM model')
        print "svm items", len(train_desc), len(train_desc[0])
        count = 0
        svm.train(np.array(train_desc), np.array(train_labels))
        svm.save(svm_model)

    # get full list of all training images
    print('Predicting test data..')
    predictions = []
    truths = []
    testing_names = os.listdir(test_dir_path)
    num_dirs = len(testing_names)
    bar = progressbar.ProgressBar(maxval=num_dirs).start()
    for i in range(num_dirs):
        p = testing_names[i]
        subdir = os.path.join(test_dir_path, p)
        images = os.listdir(subdir)
        for imgfile in images:
            imgpath = os.path.join(subdir, imgfile)
            img = cv2.imread(imgpath)
            gray = cv2.cvtColor(img, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            feature = bowDiction.compute(gray, sift.detect(gray))
            predictions.append(svm.predict(feature))
            truths.append(i)
        bar.update(i)
    bar.finish()

    acc = accuracy_score(truths, predictions)
    print('Accuracy: {}'.format(acc))
    print('Confusion matrix: ')
    cm = confusion_matrix(truths, predictions)
    print(cm)
