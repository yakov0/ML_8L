import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import warnings
import sys
import os
from random import uniform
from sklearn import svm
from random import randint

#2
data = pd.read_csv('iris.data', header=None)
#3
X = data.iloc[:, :4].to_numpy()
labels = data.iloc[:, 4].to_numpy()
#4
le = preprocessing.LabelEncoder()
Y = le.fit_transform(labels)
#5
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

#Линейный дискриминантный анализ
#1
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
clf = LinearDiscriminantAnalysis()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print(np.count_nonzero(y_test != y_pred))
#2
print(clf.fit(X_train, y_train).score(X_test, y_test) * 100)
#3
size = 0
list_test_size = []
percentage_misclassified_observations = []
classification_accuracy = []
while size <= 0.95:
    size += 0.05
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size)
    gnb = LinearDiscriminantAnalysis()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    list_test_size.append(size)
    percentage_misclassified_observations.append(np.count_nonzero(y_test != y_pred) / len(y_pred))
    classification_accuracy.append(gnb.fit(X_train, y_train).score(X_test, y_test))
fig, ax = plt.subplots()
ax.bar(list_test_size, classification_accuracy, width=0.03)
ax.bar(list_test_size, percentage_misclassified_observations, width=0.03)
ax.set_facecolor('seashell')
fig.set_figwidth(17)
fig.set_figheight(10)
fig.set_facecolor('floralwhite')
plt.show()
#4
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
size = 0
list_test_size = []
percentage_misclassified_observations = []
classification_accuracy = []
while size <= 0.95:
    size += 0.05
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size)
    gnb = LinearDiscriminantAnalysis()
    y_pred = gnb.fit(X_train, y_train).transform(X_test)
    list_test_size.append(size)
    percentage_misclassified_observations.append(np.count_nonzero(y_test != y_pred) / len(y_pred))
    classification_accuracy.append(gnb.fit(X_train, y_train).score(X_test, y_test))
fig, ax = plt.subplots()
ax.bar(list_test_size, classification_accuracy, width=0.03)
ax.bar(list_test_size, percentage_misclassified_observations, width=0.03)
ax.set_facecolor('seashell')
fig.set_figwidth(17)
fig.set_figheight(10)
fig.set_facecolor('floralwhite')
plt.show()
#5
solver_parameters = ('svd', 'lsqr', 'eigen')
for parameter in solver_parameters:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
    if parameter != 'svd':
        rand_shrinkage = uniform(0.05, 1)
        gnb = LinearDiscriminantAnalysis(solver=parameter, shrinkage=rand_shrinkage)
        y_pred = gnb.fit(X_train, y_train).predict(X_test)
        print(
            f'solver: {parameter}, shrinkage: {rand_shrinkage} - количество параметров которые неправильно были определены: {np.count_nonzero(y_test != y_pred)}')
    else:
        gnb = LinearDiscriminantAnalysis(solver=parameter)
        y_pred = gnb.fit(X_train, y_train).predict(X_test)
        print(parameter, np.count_nonzero(y_test != y_pred))
#6
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
gnb = LinearDiscriminantAnalysis(priors=[0.7, 0.15, 0.15])
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(np.count_nonzero(y_test != y_pred))
print(clf.fit(X_train, y_train).score(X_test, y_test) * 100)

#Метод опорных векторов
#1
clf = svm.SVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print(np.count_nonzero(y_test != y_pred))
print(clf.score(X, Y))
#2
print(clf.fit(X_train, y_train).score(X_test, y_test))
#3
print(clf.support_vectors_)
print(clf.support_)
print(clf.n_support_)
print('''"support_vectors_" - векторы поддержки (определение крупнейшего линейного классификатора, определяемого в характерном пространстве)
"support_" - индексы опорных векторов.
"n_support_" - количество опорных векторов для каждого класса.''')
#4
size = 0
list_test_size = []
percentage_misclassified_observations = []
classification_accuracy = []
while size <= 0.95:
    size += 0.05
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size)
    clf = svm.SVC()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    list_test_size.append(size)
    percentage_misclassified_observations.append(np.count_nonzero(y_test != y_pred) / len(y_pred))
    classification_accuracy.append(gnb.fit(X_train, y_train).score(X_test, y_test))
fig, ax = plt.subplots()
ax.bar(list_test_size, classification_accuracy, width=0.03)
ax.bar(list_test_size, percentage_misclassified_observations, width=0.03)
ax.set_facecolor('seashell')
fig.set_figwidth(17)
fig.set_figheight(10)
fig.set_facecolor('floralwhite')
plt.show()
#5
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05)

kernel_parameters = ('linear', 'poly', 'rbf', 'sigmoid')

for parameter in kernel_parameters:
    degree_rand = randint(2, 5)
    max_iter_random = randint(2, 5)

    clf = svm.SVC(kernel=parameter, degree=degree_rand, max_iter=max_iter_random)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print(
        f'''При kernel={parameter}, degree={degree_rand}, max_iter={max_iter_random}, неправильно классифицированные данные: {np.count_nonzero(y_test != y_pred)}, точность классификации: {clf.score(X, Y)}, векторы поддержки: {clf.support_vectors_}, индексы опорных векторов: {clf.support_}, количество опорных векторов для каждого класса: {clf.n_support_}''')
#6
clf = svm.NuSVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print(
    f'Неправильно классифицированные данные: {np.count_nonzero(y_test != y_pred)} точность классификации: {clf.score(X, Y)}, векторы поддержки: {clf.support_vectors_}, индексы опорных векторов: {clf.support_}, количество опорных векторов для каждого класса: {clf.n_support_}')

clf = svm.LinearSVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print(
    f'Неправильно классифицированные данные: {np.count_nonzero(y_test != y_pred)} точность классификации: {clf.score(X, Y)}')