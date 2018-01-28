'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model


model = Sequential()

model.load_weights(('first_try.h5'))

img_width, img_height = 150, 150

img = load_img('predict_1.jpg',False, (img_width, img_height))

x = img_to_array(img)

prediction = model.predict(x.reshape((1,3,img_width, img_height)),batch_size=16, verbose=0)

print(prediction)
'''

##########################################

'''
import keras
from keras.models import load_model
from keras.models import Sequential
import cv2
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
model = Sequential()

model.load_weights('first_try.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('predict_1.jpg')
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])
classes = model.predict_classes(img)
print classes
'''

##########################################

from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img


model = Sequential().load_weights('first_try.h5', by_name=True)
img = load_img('image_to_predict.jpg',False,target_size=(img_width,img_height))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = test_model.predict_classes(x)
prob = test_model.predict_proba(x)
print(preds, probs)
