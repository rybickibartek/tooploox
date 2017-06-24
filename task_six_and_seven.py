from core_functions import loadLabels, loadCNNCodes
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

def trainOrLoadSVM(X_train, y_train):
    svmFile = 'svm.pkl'
    if not os.path.exists(svmFile):
        svm = SVC(C = 0.01, kernel = 'linear', probability = True)
        svm.fit(X_train, y_train)
        pickle.dump(svm, open(svmFile, "wb"))
    return pickle.load(open(svmFile, "rb"))

def plotFreqHist(y_train, y_test, target_names):
    def prepareDataFrame(labels, target_names, name):
        df = pd.DataFrame(pd.Series(labels).value_counts()).reset_index()
        df.columns = ['class', name]
        return df.set_index(df['class'].apply(lambda val: target_names[val]))

    def merge(left, right):
        df = pd.merge(left, right, on='class')
        df.drop('class', inplace=True, axis=1)
        return df.set_index(left.index)

    dfTrain = prepareDataFrame(y_train, target_names, 'train')
    dfTest = prepareDataFrame(y_test, target_names, 'test')
    merge(dfTest, dfTrain).plot.bar()
    plt.title('Freq of classes in train and test datasets')
    plt.show()

cnnCodes = loadCNNCodes()
labels = loadLabels()
X_train, X_test, y_train, y_test = train_test_split(cnnCodes, labels, test_size=0.1, random_state=0)
target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

svm = trainOrLoadSVM(X_train, y_train)
y_pred = svm.predict(X_test)

plotFreqHist(y_train, y_test, target_names)

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred))