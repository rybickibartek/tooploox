from core_functions import getTenPercentOfData
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from skimage.feature import hog
from skimage import color
import numpy as np

def runSVM(X, y, penalty, kernel):
    clf = SVC(C = penalty, kernel = kernel)
    result = cross_val_score(clf, X, y, cv = 10)

    print('PENALTY:', penalty)
    print('KERNEL:', kernel)
    print('MEAN:', np.mean(result))
    print('STD:', np.std(result))
    print()

# solution one: NMF + SVM

def normalizeData(X):
    return Normalizer(norm='l1').fit_transform(X)

def getDecomposedAndNormalizedData(data, n_comp):
    model = NMF(n_components=n_comp, init='random', random_state=0)
    return normalizeData(model.fit_transform(data))

def solutionOne():
    data, labels = getTenPercentOfData(transformed=False)
    X = getDecomposedAndNormalizedData(data, n_comp=32)
    print('NMF + SVM')
    runSVM(X, labels, 250, 'linear')

solutionOne()

# solution two: HOG + SVM

def applyPCA(flattenHogs, pca_comp):
    pca = PCA(n_components = pca_comp)
    pca.fit(flattenHogs)
    return pca.transform(flattenHogs)

def getHoGOfImage(image):
    _, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
    return hog_image

def getHoGs(images, pca_comp = 0):

    grayImages = [color.rgb2gray(image) for image in images]
    hogs =  [getHoGOfImage(img) for img in grayImages]

    flattenHogs = np.array([img.flatten() for img in hogs])
    if pca_comp != 0:
        return applyPCA(flattenHogs, pca_comp)

    return flattenHogs

def solutionTwo():
    images, labels = getTenPercentOfData(transformed=True)
    X = getHoGs(images, pca_comp=32)
    print('HoG + PCA + SVM')
    runSVM(X, labels, 60, 'linear')

solutionTwo()