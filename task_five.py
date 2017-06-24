from core_functions import buildFeatureFilePath, loadLabels, loadCNNCodes
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

def applyTSNEToCnnCodes(cnnCodes, n_comp):
    model = TSNE(n_components=n_comp, init='pca', random_state=0)
    if not os.path.exists('reducedData' + str(n_comp) + '.npy'):
        np.save('reducedData' + str(n_comp), model.fit_transform(cnnCodes))
    reducedData = np.load('reducedData' + str(n_comp) + '.npy')
    if n_comp == 2:
        return reducedData[:, 0], reducedData[:, 1]
    else:
        return reducedData[:, 0], reducedData[:, 1], reducedData[:, 2]

def getColors():
    labels = loadLabels()
    colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
    return [colors[i] for i in labels]

def plot3d(x,y,z,colors):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, color = colors)

    ax.set_xlabel('t-sne 1')
    ax.set_ylabel('t-sne 2')
    ax.set_zlabel('t-sne 3')

    plt.show()

def plot2d(x,y,colors):
    plt.scatter(x,y,color=colors)
    plt.xlabel('t-sne 1')
    plt.ylabel('t-sne 2')
    plt.show()

def plotIn2D(colors, cnnCodes):
    x, y = applyTSNEToCnnCodes(cnnCodes, 2)
    plot2d(x, y, colors)

def plotIn3D(colors, cnnCodes):
    x, y, z = applyTSNEToCnnCodes(cnnCodes, 3)
    plot3d(x, y, z, colors)

def visualizeCNNCodes():
    colors = getColors()
    cnnCodes = loadCNNCodes()
    plotIn2D(colors, cnnCodes)
    plotIn3D(colors, cnnCodes)

visualizeCNNCodes()