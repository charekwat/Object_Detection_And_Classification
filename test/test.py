import streamlit as st
import shutil
import cv2
import os
from PIL import Image
import numpy as np
from mailbox import ExternalClashError
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
import io
from keras.applications.vgg16 import VGG16
# load the model
model = VGG16()
from keras.preprocessing.image import load_img
# load an image from file
from keras.preprocessing.image import img_to_array
# convert the image pixels to a numpy array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


def detect_Object():
  count = 0
  global detected
  detected = []
  while count < len(os.listdir('C:/Users/USER/PycharmProjects/ASSIGNMENT/frames')):
    image = load_img('frames/frame%d.jpg' %count, target_size=(224, 224))
    image = img_to_array(image)
    # convert the image pixels to a numpy array
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    object = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(object)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2] * 100))


    detected.append(label[1])
    #detected['frame%d' %count] = label[1]
    print('frame%d : ' %count, label[1])
    count = count + 1


detect_Object()

#print(detected)
#print(frames_with_object)


search = 'strawberry'

"""for i in range(len(detected.keys())):
  #if detected.[0] == 'strawberry':
    frame = 'frame%d' %i
    keys.append(frame)
print(keys)



"""