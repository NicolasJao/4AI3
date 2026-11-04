# 1. import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 2. assign X and y
X, y = make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=40)

# 3. look at the shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# 4. plotting scatter plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Data from make_blobs")
plt.show()

# 5. plotting scatter plot with colours
colors = ["red", "blue", "green", "black"]
plt.figure()
for i in range(4):
    plt.scatter(X[y == i, 0], X[y == i, 1], color=colors[i])
plt.title("Data from make_blobs with class colors")
plt.show()

# 6. split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

# 7. kmeans from scratch
# make a grid for the boundary
x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# put grid points in a list
grid_points = []
for row in range(xx.shape[0]):
    for col in range(xx.shape[1]):
        grid_points.append([xx[row, col], yy[row, col]])
grid_points = np.array(grid_points)

# pick 4 random points as the first centroids
k = 4
np.random.seed(0)
random_numbers = np.random.permutation(len(X_train))
centroids = []
for i in range(k):
    centroids.append(X_train[random_numbers[i]])
centroids = np.array(centroids)

# find the closest centroid to a point
def closest_centroid(point, centroids):
    best_index = 0
    best_distance = 1000000
    for i in range(len(centroids)):
        dx = point[0] - centroids[i][0]
        dy = point[1] - centroids[i][1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance < best_distance:
            best_distance = distance
            best_index = i
    return best_index

# repeat the steps
for iteration in range(100):
    # give each point to its closest centroid
    assignments = []
    for point in X_train:
        assignments.append(closest_centroid(point, centroids))
    assignments = np.array(assignments)

    # move each centroid to the mean
    new_centroids = []
    for i in range(k):
        points_in_cluster = X_train[assignments == i]
        if len(points_in_cluster) > 0:
            new_centroids.append(points_in_cluster.mean(axis=0))
        else:
            new_centroids.append(centroids[i])
    new_centroids = np.array(new_centroids)

    # stop if nothing moved
    if np.allclose(new_centroids, centroids):
        break
    centroids = new_centroids

# predict every grid point
scratch_grid_labels = []
for point in grid_points:
    scratch_grid_labels.append(closest_centroid(point, centroids))
scratch_grid_labels = np.array(scratch_grid_labels)
scratch_grid_labels = scratch_grid_labels.reshape(xx.shape)

# draw the boundary
plt.figure()
plt.contourf(xx, yy, scratch_grid_labels, alpha=0.3)
for i in range(4):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], color=colors[i])
plt.scatter(centroids[:, 0], centroids[:, 1], color="yellow", marker="X", s=200, edgecolor="black")
plt.title("Kmeans from scratch")
plt.show()

# 8. kmeans with sklearn
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X_train)

# predict every grid point
sk_grid_labels = kmeans.predict(grid_points)
sk_grid_labels = sk_grid_labels.reshape(xx.shape)

# draw the boundary
plt.figure()
plt.contourf(xx, yy, sk_grid_labels, alpha=0.3)
for i in range(4):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], color=colors[i])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="yellow", marker="X", s=200, edgecolor="black")
plt.title("Kmeans with sklearn")
plt.show()

# 9. KNN with sklearn
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# predict the test set
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("KNN accuracy on test set:", accuracy)

# predict every grid point
knn_grid_labels = knn.predict(grid_points)
knn_grid_labels = knn_grid_labels.reshape(xx.shape)

# draw the boundary
plt.figure()
plt.contourf(xx, yy, knn_grid_labels, alpha=0.3)
for i in range(4):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], color=colors[i])
plt.title("KNN classifier")
plt.show()

# 10. difference between Kmeans and KNN
# kmeans has no labels and makes groups
# knn uses labels and predicts a class
print("Kmeans is unsupervised. It does not use labels and it makes k groups.")
print("KNN is supervised. It uses labels and it predicts a class from the k closest points.")
print("The k in kmeans is the number of groups. The k in KNN is the number of neighbors.")
