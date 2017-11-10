

import matplotlib.pyplot as plt
import csv
# from PIL import Image
import numpy as np

def readTrafficSigns(path = "data/train"):
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = path + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(int(row[7])) # the 8th coltrainumn is the label
        gtFile.close()
    return images, labels

# X_train, y_train = readTrafficSigns("data/train")
# print(len(X_train))
# X_test, y_test = readTrafficSignsTest("data/test")
# print(X_test[100].shape)
# print(y_test)