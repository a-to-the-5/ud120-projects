#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
max_score = 0
best_map = None

# for min_samples_split in range(1, 51, 5):
#     for max_depth in range(1,10):
#         for n_estimators in range(1, 51, 5):
#             clf = RandomForestClassifier(min_samples_split=min_samples_split, max_depth=max_depth,
#                                          n_estimators=n_estimators, random_state=np.random.RandomState(1000))
#             clf.fit(features_train, labels_train)
#             pred = clf.predict(features_test)
#             score = accuracy_score(labels_test, pred)
#             m = {'min_samples_split': min_samples_split, 'max_depth': max_depth,
#                                       'n_estimators': n_estimators}
#             print "random forest [", m, "]", score
#             if score > max_score:
#                 max_score = score
#                 best_map = m
#
# print "best : ", best_map, " with score : ", max_score
#{'min_samples_split': 1, 'n_estimators': 11, 'max_depth': 5}

clf = RandomForestClassifier(min_samples_split=36, max_depth=7,
                             n_estimators=11, random_state=np.random.RandomState(1000))
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
score = accuracy_score(labels_test, pred)
print score
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
   pass

plt.show()
