#import files
import csv

from preliminar.configuration import *


# TensorFlow and tf.keras
from tensorflow import keras
from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import img_to_array
from keras.utils import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths

# Commonly used modules
import statistics
import pathlib

# Images, plots, display, and visualization
import matplotlib
#matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2

import os


#Initialize list of images, class label, bounding box coordinates, image paths
print("Loading datasets...")
data = []
labels = []
bboxes = []
imagePaths = []

#Load annotation file
with open(ANNOTATION_TRAIN, "r") as csvfile:
    rows = csv.reader(csvfile)
    next(rows)
    #Loop rows
    for row in rows:
        #Obtain each data from the csv
        #row = row.split(",")
        (w, h, startX, startY, endX, endY, label, relativeFilePath) = row

        #Reading complete filepaths and images in OpenCV format
        imagePath = os.path.sep.join([BASE_IN, relativeFilePath])
        image = cv2.imread(imagePath)

        # scale the bounding box coordinates relative to the spatial
        # dimensions of the input image
        startX = float(startX) / float(w)
        startY = float(startY) / float(h)
        endX = float(endX) / float(w)
        endY = float(endY) / float(h)

        # load the image and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        #debug for future
        # plt.imshow(image.astype(np.uint8))
        # plt.savefig(BASE_IN+"/img.jpg")
#
#
# # VGC-16
#
# base_model = VGG16(input_shape=(224, 224, 3),  # Shape of our images
#                    include_top=False,  # Leave out the last fully connected layer
#                    weights='imagenet')
#
# for layer in base_model.layers:
#     layer.trainable = False
#
# # Flatten the output layer to 1 dimension
# x = layers.Flatten()(base_model.output)
#
# # Add a fully connected layer with 512 hidden units and ReLU activation
# x = layers.Dense(512, activation='relu')(x)
#
# # Add a dropout rate of 0.5
# x = layers.Dropout(0.5)(x)
#
# # Add a final sigmoid layer with 1 node for classification output
# x = layers.Dense(1, activation='sigmoid')(x)
#
# model = tf.keras.models.Model(base_model.input, x)
#
# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
#
# vgghist = model.fit(tf_image_generator_train, validation_data=tf_image_generator_val, steps_per_epoch=100, epochs=10)