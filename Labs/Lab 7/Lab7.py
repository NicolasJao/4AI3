# 1. Importing packages
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 2. Generating the dataset
X, y = make_moons(
    n_samples=400,
    noise=0.15,
    random_state=67
)

# 3. Exploring the dataset
print(X[:5])
print(y[:5])

# The dataset has 2 features and 2 classes (0 and 1)
print("Number of features:", X.shape[1])
print("Number of classes:", len(set(y)))

# 4. Visualizing the dataset
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1')
plt.title('Make Moons Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 5. Splitting the dataset (60% train, 20% validation, 20% test)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=67)
print("Training set size:", X_train.shape[0])
print("Validation set size:", X_val.shape[0])
print("Testing set size:", X_test.shape[0])

# 6. Standardizing the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 7. Training the three SVM classifiers
linearSVM_clf = SVC(kernel='linear', C=1)
polynomialSVM_clf = SVC(kernel='poly', degree=3, C=1)
rbfSVM_clf = SVC(kernel='rbf', C=1)
linearSVM_clf.fit(X_train, y_train)
polynomialSVM_clf.fit(X_train, y_train)
rbfSVM_clf.fit(X_train, y_train)
linear_val = linearSVM_clf.score(X_val, y_val)
poly_val = polynomialSVM_clf.score(X_val, y_val)
rbf_val = rbfSVM_clf.score(X_val, y_val)
print("Linear validation accuracy:", linear_val)
print("Polynomial validation accuracy:", poly_val)
print("RBF validation accuracy:", rbf_val)

# 8. Evaluating the best model
if linear_val >= poly_val and linear_val >= rbf_val:
    best_model = linearSVM_clf
    best_name = "Linear"
elif poly_val >= rbf_val:
    best_model = polynomialSVM_clf
    best_name = "Polynomial"
else:
    best_model = rbfSVM_clf
    best_name = "RBF"
best_val = best_model.score(X_val, y_val)
test_acc = best_model.score(X_test, y_test)
print("Best model:", best_name)
print("Best model validation accuracy:", best_val)
print("Best model testing accuracy:", test_acc)

# (a) Which SVM kernel achieved the highest validation accuracy?
# The RBF kernel did, with a validation accuracy of 0.975. The linear and polynomial kernels both scored 0.8625.

# (b) How does the testing accuracy compare with the validation accuracy?
# They are the same, 0.975 for both. This means the RBF model generalizes well to data it hasn’t seen, and it didn’t just fit the validation set by chance.

# (c) Why might a nonlinear SVM kernel perform better than a linear kernel on the make_moons dataset?
# The two moons are curved, interlocking shapes, so no straight line can separate them. A linear kernel can only draw a straight boundary, which is why it scored lower (0.8625). Nonlinear kernels like RBF can bend the decision boundary to follow the curves of the moons, so it separated the classes much better (0.975).
