import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler, TerminateOnNaN, LambdaCallback
from keras import Model, optimizers
from keras import regularizers

import archs
from metrics import *
from scheduler import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



BATCH_SIZE = 64
EPOCHS = 5000
IMG_W = 128
IMG_H = 128
IMG_CHNL = 3
NUM_CLASSES = 10
LR = 1e-2
MIN_LR = 1e-3
WEIGHT_DECAY = 1e-3


def load_data():
    X = np.load('data/128x128x3_image.npy',allow_pickle=True)
    y = np.load('data/label.npy',allow_pickle=True)

    le = preprocessing.LabelEncoder()
    le.fit(y)
    # print(le.classes_)

    y = le.transform(y)
    # print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # np.savez_compressed('celebrity-faces-dataset.npz', X_train, X_test, y_train, y_test)

    return (X_train,y_train) , (X_test, y_test)


(X, y), (X_test, y_test) = load_data()

print("Training Sample: ", X.shape, " || ", "Training Labels: ", y.shape)
print("Testing Sample: ", X_test.shape, " || ", "Testing Labels: ", y_test.shape)

# for grayscale
# X = X[:, :, :, np.newaxis].astype('float32') / 255
# X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255

#for RGB Image
X = X.astype('float32') / 255
X_test = X_test.astype('float32') / 255

print("----------NORMALIZED & RESHAPED----------")
print("Training Sample: ", X.shape, " || ", "Training Labels: ", y.shape)
print("Testing Sample: ", X_test.shape, " || ", "Testing Labels: ", y_test.shape)

y = keras.utils.to_categorical(y, 10)
y_test = keras.utils.to_categorical(y_test, 10)


def build_model():
    input = Input(shape=(IMG_W, IMG_H, IMG_CHNL))
    label = Input(shape=(NUM_CLASSES,))

    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)


    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)


    x = Flatten() (x)
    x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x) # kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    x = Dropout(0.5)(x)
    output = ArcFace(NUM_CLASSES, regularizer=regularizers.l2(WEIGHT_DECAY))([x, label]) # regularizer=regularizers.l2(WEIGHT_DECAY)

    model = Model([input, label], output)

    return model

model = build_model()

model.summary()

# optimizer = SGD(lr=LR, momentum=0.5)
# optimizer = Adam(lr=LR, decay=0.004)

CALL_BACKS = [ModelCheckpoint('models/model.hdf5',
                     verbose=1, save_best_only=True)
                     ] # CosineAnnealingScheduler(T_max=EPOCHS, eta_max=LR, eta_min=MIN_LR, verbose=1)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=LR, momentum=0.5),
              metrics=['accuracy'])


model.fit([X, y],
          y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=([X_test, y_test], y_test),
          callbacks= CALL_BACKS)



model.load_weights('models/model.hdf5')
score = model.evaluate([X_test, y_test], y_test, verbose=1)

print("Test loss:", score[0])
print("Test accuracy:", score[1])
