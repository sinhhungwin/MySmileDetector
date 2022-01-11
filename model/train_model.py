# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
# from pyimagesearch.nn.conv import LeNet
from imutils import paths
import imutils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse tha arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to the input dataset of faces')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []

# loop over the input images
# Example imagePath in list : SMILEs\positives\positives7\872.jpg
for imagePath in sorted(list(paths.list_images(args['dataset']))):
    # load the image, pre-process it, and store in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-3]
    label = 'smiling' if label == 'positives' else 'not_smiling'
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# sklearn Preprocessing
# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)

# TODO: Cá nhân mình thấy dòng này thừa thãi
labels = np_utils.to_categorical(le.transform(labels), 2)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# initialize the model
print('[INFO] compiling model...')

# LeNet architecture that will accept 28×28 single channel images.
# Given that there are only two classes (smiling versus not smiling), we set classes=2
model = Sequential()

height, width, depth, classes = 28, 28, 1, 2

inputShape = (height, width, depth)

# if we are using 'channels first', update the input shape
if K.image_data_format() == 'channels_first':
    inputShape = (depth, height, width)

# first set of CONV => ReLU => POOL layers
model.add(Conv2D(20, (5, 5), padding='same', input_shape=inputShape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second layer of CONV => ReLU => POOL layers
model.add(Conv2D(50, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# first (and only) set of FC => ReLU layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

# softmax classifier
model.add(Dense(classes))
model.add(Activation('softmax'))

# model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss=['binary_crossentropy'], optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())

# train the network
print('[INFO] training network...')
# TODO đổi tên H
print(classWeight)
print('classWeight datatype')
print(type(classWeight))

classWeight = dict(enumerate(reversed(classWeight), 0))
print(classWeight)

H = model.fit(trainX, trainY, validation_data=(testX, testY),
              class_weight=classWeight,
              batch_size=64, epochs=15, verbose=1)

# evaluate the network
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print('[INFO] serializing network')
model.save(args['model'])

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 15), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 15), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 15), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, 15), H.history['val_accuracy'], label='val_accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
