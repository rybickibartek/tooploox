import numpy as np
import pickle
import os
import tarfile
import pickle
from scipy.misc import toimage

def extract(fileName = 'cifar-10.tar.gz'):
    tar = tarfile.open(fileName)
    tar.extractall()
    tar.close()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getTenPercentOfData(transformed=True):

    def transformData(data):
        resultData = []
        for i in range(len(data)):
            resultData.append(np.transpose(data[i].reshape(3, 32, 32), (1, 2, 0)))
        return resultData

    pathToBatchOne = 'cifar-10-batches-py/data_batch_1'

    if not os.path.exists(pathToBatchOne):
        extract()

    batchOne = unpickle(pathToBatchOne)
    data = batchOne[b'data'][:6000, :]
    labels = batchOne[b'labels'][:6000]

    if transformed:
        data = transformData(data)

    return data, labels

def buildFeatureFilePath(path, begin):
    return path + str(begin) + '_' + str(begin + 100)

def loadLabels(path = 'images/labels.pkl'):
    return pickle.load(open(path, 'rb'))

def loadCNNCodes(begin = 0, end = 6000, step = 100, pathToResults = 'features/'):
    features = []
    for begin in range(begin, end, step):
        val = np.load(buildFeatureFilePath(pathToResults, begin) + '.npy')
        features.append(val)

    return np.concatenate(features, axis = 0)