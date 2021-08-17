import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
import math
# Model
Classifier = Sequential()

Classifier.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same',input_shape=(64,64,3))) #64x64x32
Classifier.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu')) #64x64x64
Classifier.add(MaxPooling2D(pool_size=(2,2))) #32x32x64

Classifier.add(Conv2D(64, kernel_size=(3,3), activation='relu')) #30x30x64
Classifier.add(Conv2D(64, kernel_size=(3,3), activation='relu')) #28x28x64
Classifier.add(MaxPooling2D(pool_size=(2,2))) #14x14x64

Classifier.add(Conv2D(64, kernel_size=(3,3), activation='relu')) #12x12x64
Classifier.add(Conv2D(128, kernel_size=(3,3), activation='relu')) #10x10x128
Classifier.add(MaxPooling2D(pool_size=(2,2))) #5x5x128

Classifier.add(Conv2D(128, kernel_size=(3,3), activation='relu')) #3x3x128

Classifier.add(Flatten()) #1152

Classifier.add(Dense(500, activation='relu'))
Classifier.add(Dropout(0.3))
Classifier.add(Dense(300, activation='relu'))
Classifier.add(Dropout(0.5))
Classifier.add(Dense(5, activation='softmax'))

# Compile the Model
Classifier.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

# Fitting the CNN MODEL to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2)

valid_datagen = ImageDataGenerator(rescale=1./255.)

training_set = train_datagen.flow_from_directory('Flowers Train_Valid_Test/train',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
                                                 shuffle=True)

valid_set = valid_datagen.flow_from_directory('Flowers Train_Valid_Test/val',
                                              target_size = (64,64),
                                              batch_size = 32,
                                              class_mode = 'categorical',
                                              classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
                                              shuffle=True)

batch_size=32
trainingsize = 3019
validate_size = 860

def calculate_spe(y):
  return int(math.ceil((1. * y) / batch_size))

Steps_per_epoch = calculate_spe(trainingsize)
Validation_steps = calculate_spe(validate_size)

model = Classifier.fit_generator(training_set,
                                 steps_per_epoch= Steps_per_epoch,
                                 epochs=50,
                                 validation_data=valid_set,
                                 validation_steps=Validation_steps)

Classifier.save("Model_Flower.h5")
print("Saved Model to disk")