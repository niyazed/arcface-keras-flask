import numpy as np
import keras

from keras.models import load_model
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

IMG_W = 128
IMG_H = 128

# Encoders
in_encoder = Normalizer()
out_encoder = LabelEncoder()

# Label Encoding
labels = np.load('data/5c-face-label.npy',allow_pickle=True)
out_encoder.fit(labels)


arcface_model = load_model('models/emb-model.h5')
arcface_model.summary()

# DENSE_MODEL LOAD FOR PREDICTION
dense_model = load_model('models/5c-dense.h5')
dense_model.summary()


def get_embedding(samples):
    emd_features = arcface_model.predict(samples, verbose=1)
    emd_features /= np.linalg.norm(emd_features, axis=1, keepdims=True)
    
    return emd_features



def face_recognition(img):
    print(img.shape)
    
    # Read and process image for model
    face_img = img.astype('float32') / 255
    img = cv2.resize(face_img, (IMG_W,IMG_H))

    img = np.reshape(img, (1,img.shape[0], img.shape[1], img.shape[2]))
    img_emd = np.asarray(get_embedding(img)) # converting embedded image to numpy array if needed


    # Predicting Image
    img_norm = in_encoder.transform(img_emd)
    yhat_class = dense_model.predict_classes(img_norm)
    yhat_prob = dense_model.predict_proba(img_norm)

    # Reverse Transform to Original label
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)

    return predict_names[0], class_probability

   