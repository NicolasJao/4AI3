import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# 1. Load dataset
df = pd.read_csv("balance-scale.data", header=None, names=["Class Name", "Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"])

# 2. Merge left and right columns
df["LeftBalance"] = df["Left-Weight"] * df["Left-Distance"]
df["RightBalance"] = df["Right-Weight"] * df["Right-Distance"]

# 3. Drop columns
df = df.drop(columns=["Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"])

# 4. Visualize the data
class_labels = ["R", "B", "L"]
class_colors = ["blue", "red", "green"]
class_names = ["Right", "Balanced", "Left"]
for i, lab in enumerate(class_labels):
    subset = df[df["Class Name"] == lab]
    plt.scatter(subset["RightBalance"], subset["LeftBalance"], c=class_colors[i], label=class_names[i])
plt.xlabel("RightBalance")
plt.ylabel("LeftBalance")
plt.legend()
plt.show()

# 5. Split dataset
X = df[["RightBalance", "LeftBalance"]].values
y = df["Class Name"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

# 6. Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Visualize function
def visualize(X, y, model, title=""):
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    x_values = np.arange(x_min, x_max, 0.02)
    y_values = np.arange(y_min, y_max, 0.02)
    xx, yy = np.meshgrid(x_values, y_values)

    grid_points = np.column_stack((xx.flatten(), yy.flatten()))
    predictions = model.predict(grid_points)

    numbers = []
    for p in predictions:
        if p == "L":
            numbers.append(0)
        elif p == "B":
            numbers.append(1)
        else:
            numbers.append(2)
    Z = np.array(numbers).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, levels=[-0.5, 0.5, 1.5, 2.5], colors=["green", "red", "blue"])

    plt.scatter(X[y == "L", 0], X[y == "L", 1], c="green", label="Left")
    plt.scatter(X[y == "B", 0], X[y == "B", 1], c="red", label="Balanced")
    plt.scatter(X[y == "R", 0], X[y == "R", 1], c="blue", label="Right")

    plt.xlabel("RightBalance")
    plt.ylabel("LeftBalance")
    plt.title(title)
    plt.legend()
    plt.show()

# 8a. Train Logistic Regression models
lbfgs_model = LogisticRegression(solver="lbfgs", max_iter=1000)
liblinear_model = OneVsRestClassifier(LogisticRegression(solver="liblinear", max_iter=1000))
saga_model = LogisticRegression(solver="saga", max_iter=1000)

lbfgs_model.fit(X_train, y_train)
liblinear_model.fit(X_train, y_train)
saga_model.fit(X_train, y_train)

print("lbfgs test score:", lbfgs_model.score(X_test, y_test))
print("liblinear test score:", liblinear_model.score(X_test, y_test))
print("saga test score:", saga_model.score(X_test, y_test))

# 8b. Visualize results
visualize(X_test, y_test, lbfgs_model, "Logistic Regression (lbfgs)")
visualize(X_test, y_test, liblinear_model, "Logistic Regression (liblinear)")
visualize(X_test, y_test, saga_model, "Logistic Regression (saga)")

# 9a. Cross-validation
lbfgs_cv = cross_val_score(lbfgs_model, X_train, y_train, cv=5).mean()
liblinear_cv = cross_val_score(liblinear_model, X_train, y_train, cv=5).mean()
saga_cv = cross_val_score(saga_model, X_train, y_train, cv=5).mean()

print("lbfgs average CV score:", lbfgs_cv)
print("liblinear average CV score:", liblinear_cv)
print("saga average CV score:", saga_cv)

# 9b. Model selection
if lbfgs_cv >= liblinear_cv and lbfgs_cv >= saga_cv:
    print("Best model: lbfgs")
    print("Justification: lbfgs has the highest average 5-fold CV score", lbfgs_cv, "and a test score of", lbfgs_model.score(X_test, y_test))
elif liblinear_cv >= saga_cv:
    print("Best model: liblinear")
    print("Justification: liblinear has the highest average 5-fold CV score", liblinear_cv, "and a test score of", liblinear_model.score(X_test, y_test))
else:
    print("Best model: saga")
    print("Justification: saga has the highest average 5-fold CV score", saga_cv, "and a test score of", saga_model.score(X_test, y_test))
