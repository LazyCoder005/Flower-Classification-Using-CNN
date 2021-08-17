'''
author: AaryaPanchal
'''

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

class Flowers:
    def __init__(self,filename):
        self.filename = filename

    def prediction_on_flowers(self):
        #load model
        model = load_model('Model_Flower.h5')

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'daisy'
            return [{"Image": prediction}]
        elif result[0][1] == 1:
            prediction = 'dandelion'
            return [{"Image": prediction}]
        elif result[0][2] == 1:
            prediction = 'rose'
            return [{"Image": prediction}]
        elif result[0][3] == 1:
            prediction = 'sunflower'
            return [{"Image": prediction}]
        else:
            prediction = 'tulip'
            return [{"Image": prediction}]