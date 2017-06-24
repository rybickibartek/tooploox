from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

def getFeatureExtractor():
    model = InceptionV3(weights='imagenet')
    return Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)