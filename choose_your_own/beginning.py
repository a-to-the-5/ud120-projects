# SVM (rbf, C = 10000)                  :    0.990898748578
# Naive Bayes                           :    0.973265073948
# Decision Tree (min_samples_split=40)  :    0.977815699659
# AdaBoost                              :    0.950511945392  in  70.760999918 s
# Random forest                         :    0.994311717861  in  4.91600012779 s
# KNN                                   :    0.857224118316  in  173.974999905 s

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

t1 = time()
ada_boost = AdaBoostClassifier()
ada_boost.fit(features_train, labels_train)
pred = ada_boost.predict(features_test)
print "AdaBoost achieved: ", accuracy_score(labels_test, pred), " in ", (time() - t1), "s"

t1 = time()
random_forest = RandomForestClassifier()
random_forest.fit(features_train, labels_train)
pred = random_forest.predict(features_test)
print "Random forest achieved: ", accuracy_score(labels_test, pred), " in ", (time() - t1), "s"

t1 = time()
knn = KNeighborsClassifier()
knn.fit(features_train, labels_train)
pred = knn.predict(features_test)
print "KNN achieved: ", accuracy_score(labels_test, pred), " in ", (time() - t1), "s"

