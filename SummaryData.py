from LoadData import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
X , y = readTrafficSigns()
X_train, X_tc, y_train, y_tc = train_test_split(X,y,test_size=0.3,random_state=42,shuffle=True)
X_test, X_valid,y_test, y_valid = train_test_split(X_tc, y_tc, test_size=0.3,shuffle=True)

# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_valid = len(X_valid)

# Number of testing examples.
n_test = len(X_test)

print("Size of training set =", n_train)
print("Size of validation set =", n_valid)
print("Size of testing set =", n_test)

# TODO: What's the shape of an traffic sign image?
sample_image = X_train[0].squeeze()
image_shape = sample_image.shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.unique(y_train).size

print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

plt.hist(y_train, bins=np.arange(y_train.min(), y_train.max() + 1))
plt.show()