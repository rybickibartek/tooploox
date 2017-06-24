from core_functions import getTenPercentOfData
import matplotlib.pyplot as plt
from scipy.misc import toimage
import matplotlib.gridspec as gridspec
import numpy as np

def getImages(data, labels):

    def getIndexes():
        stillRequiredClasses = [i for i in range(10)]
        interestingIndexes = [[] for i in stillRequiredClasses]
        for index, value in enumerate(labels):
            if value in stillRequiredClasses:
                interestingIndexes[value].append(index)
                if len(interestingIndexes[value]) == 10:
                    stillRequiredClasses.remove(value)
                    if len(stillRequiredClasses) == 0:
                        break
        return np.array(interestingIndexes).flatten()

    return [toimage(data[i]) for i in getIndexes()]

def plotImages(data, labels):
    images = getImages(data, labels)
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows = 10, ncols = 10, wspace=0.0)
    ax = [plt.subplot(gs[i]) for i in range(100)]
    gs.tight_layout(fig, h_pad=-2.5,w_pad=-3.0)

    for i,im in enumerate(images):
        ax[i].imshow(im)
        ax[i].axis('off')
    plt.show()

data, labels = getTenPercentOfData()
plotImages(data, labels)