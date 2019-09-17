import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras import Model, optimizers
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.optimizers import SGD, Adam

import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.externals import joblib

import archs
from metrics import *


BATCH_SIZE = 128
EPOCHS = 10000
IMG_W = 128
IMG_H = 128
IMG_CHNL = 3
LR = 1e-2
NUM_CLASSES = 10
WEIGHT_DECAY = 1e-3

# Encoders
in_encoder = Normalizer()
out_encoder = LabelEncoder()

# Label Encoding
labels = np.load('data/label.npy',allow_pickle=True)
out_encoder.fit(labels)

# SVM_MODEL LOAD FOR PREDICTION
model = joblib.load('models/128_svm_model.sav')



def build_model():
    
    
    input = Input(shape=(IMG_W, IMG_H, IMG_CHNL))
    label = Input(shape=(NUM_CLASSES,))

    x = Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)


    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x) # kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    x = BatchNormalization()(x)
    
    output = ArcFace(NUM_CLASSES, regularizer=regularizers.l2(WEIGHT_DECAY))([x, label]) # regularizer=regularizers.l2(WEIGHT_DECAY)

    model = Model([input, label], output)

    model.load_weights('models/128_model.hdf5')

    model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr=LR, momentum=0.5),
                metrics=['accuracy'])

    return model



def get_embedding(samples):
    af_model = build_model()
    af_model = Model(inputs=af_model.input[0], outputs=af_model.layers[-3].output)
    emd_features = af_model.predict(samples, verbose=1)
    emd_features /= np.linalg.norm(emd_features, axis=1, keepdims=True)
    
    return emd_features



def face_recognition(img):
    # Read and process image for model
    # face_img = cv2.imread(img, 1)
    face_img = img.astype('float32') / 255
    rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb_img, (IMG_W,IMG_H))

    img = np.reshape(img, (1,img.shape[0], img.shape[1], img.shape[2]))
    img_emd = np.asarray(get_embedding(img)) # converting embedded image to numpy array if needed


    # Predicting Image
    img_norm = in_encoder.transform(img_emd)

    yhat_class = model.predict(img_norm)
    yhat_prob = model.predict_proba(img_norm)

    # Reverse Transform to Original label
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)

    return predict_names[0], class_probability