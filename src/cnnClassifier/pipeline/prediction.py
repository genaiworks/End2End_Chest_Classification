import numpy as np 
import tensorflow as tf
# from tensorflow.keras.model import load_model
# from tensorflow.keras.preprocessing import image
import keras
from keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import keras.utils as image
# from keras.preprocessing import image
from keras.utils import load_img, img_to_array
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename=filename
    
    def predict(self):
        model = load_model(os.path.join("artifacts","training","model.h5"))

        imagename =self.filename
        test_image = load_img(imagename, target_size=(224,224))
        test_image = img_to_array(test_image)
        #it takes the batch dimension therefore expanding the dimension
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = "Normal"
            return [{"image" : prediction}]
        else:
            prediction = "Not normal"
            return [{"image" : prediction}]
