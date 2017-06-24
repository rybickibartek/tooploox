import urllib.request
from core_functions import extract, getTenPercentOfData
import numpy as np
from scipy.misc import toimage
import pickle
import shutil
import os

def loadAndExtractCifarTen(fileName):
    if not os.path.exists(fileName):
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", fileName)
    extract(fileName)

def removeFolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def removeAndCreateFolder(path):
    removeFolder(path)
    os.makedirs(path)

def saveImages(data, labels, path = 'images/'):
    removeAndCreateFolder(path)
    for index, content in enumerate(data):
        toimage(content).save(path + str(index) + '.jpg', 'JPEG')
    pickle.dump(labels, open(path + 'labels.pkl', "wb"))

fileName = 'cifar-10.tar.gz'
loadAndExtractCifarTen(fileName)
data, labels = getTenPercentOfData()
saveImages(data, labels)