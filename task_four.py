from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from core_functions import buildFeatureFilePath
from vgg_model import getFeatureExtractor
import numpy as np
import os

def createFolderIfNotExists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def loadImages(fileNames):
    images = []
    for fileName in fileNames:
        temp_img = image.load_img(fileName, target_size=(299, 299))
        images.append(image.img_to_array(temp_img))
    return preprocess_input(np.array(images))

def getImagesNames(begin, end):
    return ['images/' + str(f) + '.jpg' for f in range(begin, end)]

def computeAndSaveCNNCodes(pathToResults = 'features/'):
    featureExtractor = getFeatureExtractor()
    createFolderIfNotExists(pathToResults)

    for begin in range(0, 6000, 100):
        fileName = buildFeatureFilePath(pathToResults, begin)
        if not os.path.exists(fileName + '.npy'):
            images = loadImages(getImagesNames(begin, begin + 100))
            np.save(fileName, featureExtractor.predict(images))

        if not (begin + 100) % 300:
            print('Done:', (begin+100) / 60, '%')

computeAndSaveCNNCodes()