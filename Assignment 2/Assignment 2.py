# 4AI3 Assignment 2
# Nicolas Jao
# 400450151

from random import seed
from random import randrange
from sklearn import datasets
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Section 1
def euclidean(point_1, point_2):
    distance = 0
    for i in range(len(point_1)):
        distance += (point_1[i] - point_2[i]) ** 2
    return distance ** 0.5

seed(7)

def random_list(dimension):
    point = []
    for i in range(dimension):
        point.append(randrange(0, 51, 2))
    return point

pt_1 = random_list(2)
pt_2 = random_list(2)

print(euclidean(pt_1, pt_2))

def manhattan(point_1, point_2):
    distance = 0
    for i in range(len(point_2)):
        distance += abs(point_1[i] - point_2[i])
    return distance

print(manhattan(pt_1, pt_2))

# Section 2
# learning how KNN algorithm works... pretty simple grade 9 math

# Section 3
iris = datasets.load_iris()
list(iris.keys())
print(iris['DESCR'])
print(iris['feature_names'])
X = iris['data']
y = (iris['target'] == 2).astype(int)
flower_1 = X[0]
flower_2 = X[1]
print(euclidean(flower_1, flower_2))

# Section 4
def min_max_normalize(data):
    data_normalized = copy.deepcopy(data)
    for i in range(len(data[0])):
        minimum = min(data[:, i])
        maximum = max(data[:, i])
        feature_normalized = []
        for value in data[:, i]:
            value = (value - minimum) / (maximum - minimum)
            feature_normalized.append(value)
        data_normalized[:, i] = feature_normalized
    return data_normalized

X_normalized = min_max_normalize(X)
print(X_normalized[:, 0][0])

def classify(unknown, X, y, k):
    distances = []
    virginica = 0
    non_virginica = 0
    for i in range(len(X)):
        distance_to_point = euclidean(X[i], unknown)
        distances.append([distance_to_point, i])
    distances.sort()
    nearest_neighbors = distances[0:k]
    for iris in nearest_neighbors:
        index = iris[1]
        if y[index] == 1:
            virginica += 1
        elif y[index] == 0:
            non_virginica += 1
    if virginica >= non_virginica:
        result = 1
    else:
        result = 0
    return result

print(classify([.2, .6, .1, .05], X_normalized, y, 3))

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=67) # personal seed
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

print(knn_model.score(X_test, y_test))

# Step 10 and 11
best_k = 1
best_score = 0
for k in range(1, len(X_train) + 1):
    NicolasJaoModel = KNeighborsClassifier(n_neighbors=k)
    NicolasJaoModel.fit(X_train, y_train)
    score = NicolasJaoModel.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_k = k
print(best_k)
print(best_score)
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
print(final_model.score(X_test, y_test)) # redundant answer checking
