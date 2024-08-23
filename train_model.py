# Courtesy of NT

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from glob import glob

#print(tf. __version__)

#print(tf.keras. __version__)

img_rows , img_cols = 299,299
train_path = 'train_mini'
test_path = 'validation_mini'

InResV2_model =InceptionResNetV2(weights='imagenet', include_top= True, input_shape=(img_rows,img_cols,3))

InResV2_model.summary()

InResV2_model= InceptionResNetV2(weights='imagenet', include_top= False, input_shape=(img_rows,img_cols,3))

def addTopModelInResv2(bottom_model,num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1536,activation='relu')(top_model)
    top_model = Dense(256,activation='relu')(top_model)
    top_model = Dense(192,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

num_classes=3

FC_Head = addTopModelInResv2(InResV2_model, num_classes)

model=Model(inputs = InResV2_model.input,outputs= FC_Head)
print(model.summary())

from keras.preprocessing.image import ImageDataGenerator
train_data_dir='train_mini'
validation_data_dir='validation_mini'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.3,
                                   height_shift_range=0.1,
                                   horizontal_flip=True, 
                                   shear_range = 0.2,
                                   zoom_range= 0.2,
                                   fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size =1
train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_rows,img_cols),batch_size=batch_size,
                                                   class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,target_size=(img_rows,img_cols),
                                                              batch_size=batch_size,
                                                   class_mode='categorical')

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

#checkpoint = ModelCheckpoint("/home/nimisha/Documents/sample_powerAI/Rice_dataset/models/newRice_InResv2_11_26_40.h5",
                            # monitor="val_loss",mode="min",save_best_only = True,verbose=1)

model.compile(loss= 'categorical_crossentropy',optimizer = Adam(learning_rate = 0.00001), metrics = ['accuracy'])

#model.compile(optimizer = Adam(learning_rate =[1e-2,1e-3]),
                 #loss = 'categorical_crossentropy',
                 #metrics = ['accuracy'])

earlystop=EarlyStopping(monitor ='val_loss',min_delta = 0,patience = 8,verbose = 1,restore_best_weights = True)

#callbacks = [earlystop , checkpoint]

nb_train_samples = 15
nb_validation_samples = 15
epochs = 2
batch_size = 1

fit = model.fit_generator(train_generator,steps_per_epoch = nb_train_samples // batch_size, epochs = epochs,
                              #callbacks = callbacks, 
                          validation_data = validation_generator,
                              validation_steps = nb_validation_samples // batch_size)

#loss
plt.plot(fit.history['loss'], label = 'train_loss')
plt.plot(fit.history['val_loss'], label ='val_loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

#accuracy
plt.plot(fit.history['accuracy'], label='train_acc')
plt.plot(fit.history['val_accuracy'], label = 'val_acc')
plt.legend()
plt.show()
plt.savefig('AccuracyVal_acc')

import tensorflow as tf
from keras.models import load_model

model.save('copy_newRice_InResv2_11_25_40.h5')

classifier = load_model('copy_newRice_InResv2_11_25_40.h5')

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np

evaluation_data_dir= 'evaluation_mini'
evaluation_datagen = ImageDataGenerator(rescale=1./255)

evaluation_generator = evaluation_datagen.flow_from_directory(evaluation_data_dir,target_size=(img_rows,img_cols),
                                                              batch_size=batch_size,
                                                              classes=['1121','1509','1637'],
                                                   class_mode='categorical', shuffle = False)
true_classes = evaluation_generator.classes
class_labels = list(evaluation_generator.class_indices)   

class_labels

target_names = ['1121','1509','1637']
Y_pred = model.predict_generator(evaluation_generator, 440 // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = metrics.confusion_matrix(evaluation_generator.classes, y_pred)
print(cm)
print('Classification Report')
print(metrics.classification_report(evaluation_generator.classes, y_pred))

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
img1 = image.load_img('PD_1121_1_1.jpg', target_size=(299, 299))
img = image.img_to_array(img1)
img = img/255
# create a batch of size 1 [N,H,W,C]
img = np.expand_dims(img, axis=0)
prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
print(target_names[0] + ':', prediction[0][0])
print(target_names[1] + ':', prediction[0][1])
print(target_names[2] + ':', prediction[0][2])

#1637/PD_1637_8_2.jpg
#1637/PD_1637_19_2.jpg

# Courtesy of NT
