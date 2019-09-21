import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

labels = []
BASE = 'faces/'
for folder in os.listdir(BASE):
    labels.append(folder)


img_vector = []
label_vector = []

for label in labels:

    # print(label)
    images = [BASE + label + '/' + img_file for img_file in os.listdir(BASE + '/' + label)]
    
    for image in images:
        print(image)
        img = cv2.imread(image,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128,128))
        img_vector.append(img)
        label_vector.append(label)
    
np.save('data/face-image.npy', np.asarray(img_vector))
np.save('data/face-label.npy', np.asarray(label_vector))

i = np.load('data/face-image.npy',allow_pickle=True)
l = np.load('data/face-label.npy',allow_pickle=True)

print(len(i), len(l))

print(i.shape, l.shape)

plt.imshow(i[427])
print(l[427])
plt.show()



