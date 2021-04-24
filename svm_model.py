#!/usr/bin/env python3
# Ref: https://github.com/monjoybme/SVM-Image-Classification

#### load packages
import numpy as np
from pathlib import Path                          # directory path package

from skimage.io import imread                     # read image
from skimage.transform import resize              # image resize
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch                   # function return class
from sklearn.model_selection import GridSearchCV  # hyper parameter optimization using grid search


def load_image_files(container_path, dimension=(64, 64)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)                                                     # container directory
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]     # list of directories
    categories = [fo.name for fo in folders]                                             # ['cat','plane'] --> categories list

    descr = "image classifier data flattening and labeling"
    images = []                                                 # images list
    flat_data = []                                              # X_train --> flattened image for classifier input
    target = []                                                 # y_train --> rank 0 target extracted from directory
    for i, direc in enumerate(folders):                         # loop over folders
        for file in direc.iterdir():
            try:                                                # skip if .DS_store/ thunmnail directory is present
                img = imread(file)                              # read image
                img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')     # reshape image to 64 x 64
                flat_data.append(img_resized.flatten())         # append flattened image
                images.append(img_resized)                      # append resized image (not flattened)
                target.append(i)                                # append category 
            except:
                continue                                        # skip if .DS_store/ thunmnail directory is present
    
    flat_data = np.array(flat_data)                             # X_train shape (100, 12288)
    target = np.array(target)                                   # y_train shape (100,)
    images = np.array(images)                                   # (100, 64, 64, 3)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

def train_SVM(container_path = "images_svm"):
    image_dataset = load_image_files(container_path)
    
    ### search for parameters 
    param_grid = [
                  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
                 ]
    X_train, y_train = image_dataset.data, image_dataset.target   # get the train and test data

    svc = svm.SVC()                            # create SVM class
    clf = GridSearchCV(svc, param_grid, verbose=1)        # search over grid
    clf.fit(X_train, y_train)                  # train SVM

    return clf


def model_report(clf, X_test, y_test):

    y_pred = clf.predict(X_test)               # predic SVM
    print("Classification report for - \n{}:\n{}\n".format(clf, metrics.classification_report(y_test, y_pred)))  # classifier report




clf = train_SVM(container_path = "images_svm")     # train folder location SVM

image_test = load_image_files("images_test")       # test folder location   
print('Test Data shape: ', image_test.data.shape, '\nTest label shape: ', image_test.target.shape, '\n')

model_report(clf, X_test = image_test.data , y_test= image_test.target)