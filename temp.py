# -*- coding: utf-8 -*-
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""
Spyder Editor

This is a temporary script file.
"""

from scipy import io
from sklearn import preprocessing
from sklearn import svm
from sklearn import grid_search
from numpy import array
poop = io.loadmat("cs6923Project.mat")

testvar = poop.get('test')
trainvar = poop.get('train')
trainl = poop.get('train_label') 
trainl = (array(trainl)).ravel()
scale  = preprocessing.MinMaxScaler().fit(trainvar) #normalizing the data
train_scale = scale.transform(trainvar)
test_scale = scale.transform(testvar)

gird = [{'C':[0.1], 'kernel' : ['linear']}]
suppvec = svm.SVC()
suppvec1 = grid_search.GridSearchCV(suppvec, gird, cv = 10, n_jobs = 10)
print('Start1')
ydirlin = suppvec1.fit(train_scale,trainl)

print('Start')
testsetvar = []
predvar = []
j=0

for i in range(50000):
    predvar.append(ydirlin.predict(train_scale[i]))
    testsetvar.append(ydirlin.predict(test_scale[i]))
    if (ydirlin.predict(train_scale[i]) == trainl[i]):
        j += 1
print(j)
print('Done')        
        
io.savemat('train_var_predict.mat', {'predict' : predvar})
io.savemat('test_var_predict.mat', {'predict': testsetvar})
