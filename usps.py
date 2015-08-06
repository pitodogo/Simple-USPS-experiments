from __future__ import division,print_function
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB ## <- Not sure if you want
## the Gaussian one
from sklearn import tree
from scipy.io import loadmat

import numpy as np


## Load Data
train_data = []
train_label =[]
f = open("optdigits.tra","rw")
for line in f:
    train_label.append(int(line[-2]))
    splited_data = line[:-3].split(',')
    splited_data = [int(s) for s in splited_data]
    train_data.append(splited_data)
f.close()

test_data = []
test_label =[]
f = open("optdigits.tes","rw")
for line in f:
    test_label.append(int(line[-2]))
    splited_data = line[:-3].split(',')
    splited_data = [int(s) for s in splited_data]
    test_data.append(splited_data)
f.close()


## SVM

svc = SVC(C=1,class_weight='auto')
svc.fit(train_data,train_label)
svm_result = svc.predict(test_data)
diff = svm_result - test_label
err = np.count_nonzero(diff)
acc_SVM = 1 - err/len(test_label)
print("SVM accuracy:",acc_SVM)

## NN

acc_NN = 0
nn = KNeighborsClassifier(n_neighbors=1)
nn.fit(train_data,train_label)
nn_result = nn.predict(test_data)
diff = nn_result - test_label
err = np.count_nonzero(diff)
acc_NN = 1 - err/len(test_label)
print("NN accuracy:",acc_NN)

## Decision Tree

dt = tree.DecisionTreeClassifier()
dt.fit(train_data,train_label)
dt_result = dt.predict(test_data)
diff = dt_result - test_label
err = np.count_nonzero(diff)
acc_DT = 1 - err/len(test_label)
print("DT accuracry:",acc_DT)

## Navie Bayes
gnb = GaussianNB()
gnb.fit(train_data,train_label)
nb_result = gnb.predict(test_data)
diff = nb_result - test_label
err = np.count_nonzero(diff)
acc_NB = 1 -err/len(test_label)
print("NB accuracy:",acc_NB)
