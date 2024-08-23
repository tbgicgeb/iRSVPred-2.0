#!/usr/bin/python

# Courtesy of NT
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#print(sys.version_info)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image

input_dir_path = sys.argv[1]

classifier = load_model('Rice_InResv2_Aug36_25epochs.h5')
img1 = image.load_img(input_dir_path, target_size=(299, 299))
img = image.img_to_array(img1)
img = img/255
# create a batch of size 1 [N,H,W,C]
img = np.expand_dims(img, axis=0)
prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
#print(prediction)

target_names = ['1121','1509','1637','1718','1728','BAS_370','CSR_30','DHBT_3','PB1','PB_6','Unknown']
#COurtesy of NT
print(target_names[0] + ':', prediction[0][0])
print(target_names[1] + ':', prediction[0][1])
print(target_names[2] + ':', prediction[0][2])
print(target_names[3] + ':', prediction[0][3])
print(target_names[4] + ':', prediction[0][4])
print(target_names[5] + ':', prediction[0][5])
print(target_names[6] + ':', prediction[0][6])
print(target_names[7] + ':', prediction[0][7])
print(target_names[8] + ':', prediction[0][8])
print(target_names[9] + ':', prediction[0][9])
print(target_names[10] + ':', prediction[0][10])

#COurtesey of NT
