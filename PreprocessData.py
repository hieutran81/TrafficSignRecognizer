from LoadData import *
from sklearn.model_selection import train_test_split
import cv2
import datetime

def splitData():
    X, y = readTrafficSigns()
    ar = np.asarray(X)
    for i in range(0,len(ar)):
        img = ar[i]
        print(ar[i].shape)
        ar[i] = np.reshape(img,[2610])
        ar[i] = np.reshape(img,[3072])
        ar[i] = np.reshape(img,[32,32,3])
        print(ar[i].shape)

    X_train, X_tc, y_train, y_tc = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    X_test, X_valid, y_test, y_valid = train_test_split(X_tc, y_tc, test_size=0.3, shuffle=True)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def applyGrayscaleAndEqualizeHist(data):
    length = len(data)
    print("Applying Grayscale filter and Histogram Equalization")

    filteredData = []

    for data_sample in data[0:length, :]:
        grayScale = cv2.cvtColor(data_sample, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(grayScale)
        filteredData.append(np.reshape(equalized, (32, 32, 1)))

    return np.array(filteredData)

def normalize(data):
    length = len(data)
    data = data.astype(np.float, copy = False)

    print("Starting normalization: ", datetime.datetime.now().time())
    for data_sample in data[0:length, :]:
        for data_sample_row in data_sample:
            for data_sample_pixel in data_sample_row:
                data_sample_pixel[:] = [(color - 127.5) / 255.0 for color in data_sample_pixel]

    print("Normalization finished: ", datetime.datetime.now().time())
    return data

# preprocess before go to cnn
def preprocess():
    X_train, y_train, X_valid, y_valid, X_test, y_test = splitData()

    X_train = applyGrayscaleAndEqualizeHist(X_train)
    X_valid = applyGrayscaleAndEqualizeHist(X_valid)
    X_test = applyGrayscaleAndEqualizeHist(X_test)

    # Normalization of pixel values from [0,255] to [-1,1]
    X_train = normalize(X_train)
    X_valid = normalize(X_valid)
    X_test = normalize(X_test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

preprocess()



