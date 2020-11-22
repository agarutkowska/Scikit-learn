#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2)

plt.scatter(X[:,0], X[:,1], c=y, s=4) # s - rozmiar kropek 
plt.show()

classifiers = {
    "Gaussian" : GaussianNB(),
    "QDA" : QuadraticDiscriminantAnalysis(),
    "KNeighbours" : KNeighborsClassifier(),
    "SVC" : SVC(probability=True),
    "Tree" : DecisionTreeClassifier()
}

iter = 100
accuracy = []
recall = []
precision = []
F1 = []
roc = []
t_fit = []
t_predict = []
clfs = []
for i in range(iter):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for clfname, clf in classifiers.items():
        t_fit_start = time.time()
        clf.fit(X_train, y_train)
        t_fit_end = time.time()
        t_fit.append(t_fit_end - t_fit_start)
        
        t_predict_start = time.time()
        y_pred = clf.predict(X_test)
        pr_pred = clf.predict_proba(X_test)
        t_predict_end = time.time()
        t_predict.append(t_predict_end - t_predict_start)
        
        clfs.append(clfname)
        accuracy.append(metrics.accuracy_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred))
        F1.append(metrics.f1_score(y_test, y_pred))
        roc.append(metrics.roc_auc_score(y_test, pr_pred[:,-1]))
        
        if i == iter - 1:
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            
            plt.title(f'Krzywa ROC dla {clfname}')
            plt.plot(fpr, tpr, 'b', label = f'AUC = {roc_auc: 0.0f}')
            plt.legend(loc = 'lower right')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()

df = pd.DataFrame({"clfs": clfs, "Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1": F1, "ROC": roc, \
    "Training time": t_fit, "Testing time": t_predict})
df.groupby(["clfs"]).mean()

classifiers2 = {
    'Gaussian' : GaussianNB(),
#     'QDA' : QuadraticDiscriminantAnalysis(),
    'KNeighbours' : KNeighborsClassifier(),
    'SVC' : SVC(probability=True),
    'Tree' : DecisionTreeClassifier()
}

parameters = {
    'Gaussian': {
        'priors' : [[0.3, 0.7], [0.35, 0.65], [0.4, 0.6], [0.45, 0.55], [0.5, 0.5], [0.6, 0.4]],
        'var_smoothing' : [1e-2, 1e-6, 1e-8, 1e-9, 1e-10, 1e-1, 1.0]
    },
    'QDA' : {
        'priors' : [[0.1, 0.9],[0.5, 0.5],[0.8, 0.2]],
        'reg_param' : [0.0, 0.1, 0.2,],
        'store_covariance' : (False, True),
        'tol' : [1e-2, 1e-9, 1e-1, 1.0]
    },
    'KNeighbours' : {
        'n_neighbors' : [4,5,6],
        'weights' : ('uniform', 'distance'),
        'algorithm' : ('auto', 'ball_tree', 'kd_tree', 'brute'),
        'leaf_size' : [10,15,20],
        'p' : [1,2,3]
    },
    'SVC' : {
        'C' : [1, 100, 1000, 10000],
        'degree' : [2,3,4],
        'gamma' : ('scale', 'auto'),
        'coef0' : [0.0, 0.1, 0.2]
    },
    'Tree' : {
        'criterion' : ('gini', 'entropy'),
        'splitter' : ('best', 'random'),
        'max_depth' : [10, 15, 20],
        'min_samples_split' : [0.2, 0.4, 0.8, 1.0],
        'min_samples_leaf' : [1, 2, 3]
    }
}

scoring = {'AUC': 'roc_auc', 'Accuracy': metrics.make_scorer(metrics.accuracy_score)}
best = []

for clfname in classifiers2.keys():
    for paramclf in parameters.keys():
        if clfname == paramclf:
            clf_par = GridSearchCV(classifiers2[clfname], 
                                   parameters[paramclf],
                                   scoring=scoring,
                                   refit='Accuracy', 
                                   return_train_score=True)
            clf_par.fit(X, y) 
            best.append(clf_par.best_params_)

classifiers3 = {
    "Gaussian" : GaussianNB(priors=best[0]['priors'],
                           var_smoothing=best[0]['var_smoothing']),
    "KNeighbours" : KNeighborsClassifier(algorithm=best[1]['algorithm'],
                                         leaf_size=best[1]['leaf_size'],
                                         n_neighbors=best[1]['n_neighbors'],
                                         p=best[1]['p'],
                                         weights=best[1]['weights']),
    "SVC" : SVC(probability=True, 
                C=best[2]['C'],
                coef0=best[2]['coef0'],
                degree=best[2]['degree'],
                gamma=best[2]['gamma']),
    "Tree" : DecisionTreeClassifier(criterion=best[3]['criterion'],
                                    max_depth=best[3]['max_depth'],
                                    min_samples_leaf=best[3]['min_samples_leaf'],
                                    splitter=best[3]['splitter'])
}

iter = 100
accuracy = []
recall = []
precision = []
F1 = []
roc = []
t_fit = []
t_predict = []
clfs = []
for i in range(iter):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for clfname, clf in classifiers3.items():
        t_fit_start = time.time()
        clf.fit(X_train, y_train)
        t_fit_end = time.time()
        t_fit.append(t_fit_end - t_fit_start)
        
        t_predict_start = time.time()
        y_pred = clf.predict(X_test)
        pr_pred = clf.predict_proba(X_test)
        t_predict_end = time.time()
        t_predict.append(t_predict_end - t_predict_start)
        
        clfs.append(clfname)
        accuracy.append(metrics.accuracy_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred))
        F1.append(metrics.f1_score(y_test, y_pred))
        roc.append(metrics.roc_auc_score(y_test, pr_pred[:,-1]))
        
        if i == iter - 1:
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            
            plt.title(f'Krzywa ROC dla {clfname}')
            plt.plot(fpr, tpr, 'b', label = f'AUC = {roc_auc: 0.0f}')
            plt.legend(loc = 'lower right')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()

df = pd.DataFrame({"clfs": clfs, "Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1": F1, "ROC": roc, \
    "Training time": t_fit, "Testing time": t_predict})
df.groupby(["clfs"]).mean()
