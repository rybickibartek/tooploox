from core_functions import loadLabels, loadCNNCodes
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import xgboost as xgb

def trainOrLoadXGB(X_train, y_train, xgb_params):
    xgbFile = 'xgb.pkl'
    if not os.path.exists(xgbFile):
        model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=position, feval=r_score, maximize=True)
        pickle.dump(model, open(xgbFile, "wb"))
    return pickle.load(open(xgbFile, "rb"))

cnnCodes = loadCNNCodes()
labels = loadLabels()

target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
X_train, X_test, y_train, y_test = train_test_split(cnnCodes, labels, test_size=0.1, random_state=0)

clf = RandomForestClassifier(n_estimators=300)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred))