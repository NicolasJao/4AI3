# 2. Import the related packages
import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# 3. Load the dataset and verify that the required headers exist
waterfalls = pandas.read_csv("City_Waterfalls.csv")
required_headers = [
    "LONGITUDE", "LATITUDE", "NAME", "ACCESS_FROM", "HEIGHT_IN_M",
    "WIDTH_IN_M", "OWNERSHIP", "RANKING", "TYPE",
]
print("Header check:")
for header in required_headers:
    status = "OK" if header in waterfalls.columns else "MISSING"
    print(f"{header:15s} -> {status}")
missing = [h for h in required_headers if h not in waterfalls.columns]
if missing:
    print("WARNING: These headers were not found:", missing)
    print("Columns actually in the file:", list(waterfalls.columns))
print("First five rows:")
print(waterfalls.head())

# 4. Data types and statistical summary
print("Checking data type of each feature in the dataset:")
waterfalls.info()
print("Statistical summary of the numerical features in the dataset:")
print(waterfalls.describe())

# 5. Checking which community in Hamilton has the most waterfalls
# ---------------------------------------------------------------------------
print("Waterfalls per community:")
community_counts = waterfalls["COMMUNITY"].value_counts()
print(community_counts)
print(f"Community with the most waterfalls: {community_counts.idxmax()} "
      f"({community_counts.max()} waterfalls)")

# 6. Distribution of the RANKING feature
print("RANKING distribution:")
print(waterfalls["RANKING"].value_counts())

# 7. Select certain features from the original DataFrame
selected_cols = ["NAME", "LONGITUDE", "LATITUDE", "HEIGHT_IN_M", "WIDTH_IN_M", "OWNERSHIP"]
selected = waterfalls[selected_cols].copy()
print("Selected features: ")
print(selected.head())

# 8. Scatter plot of longitude vs. latitude
plt.figure(figsize=(7, 6))
plt.scatter(selected["LONGITUDE"], selected["LATITUDE"], alpha=0.7)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Locations of Hamilton Waterfalls")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 9. Histograms of HEIGHT_IN_M and WIDTH_IN_M
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(selected["HEIGHT_IN_M"].dropna(), bins=15, edgecolor="black")
axes[0].set_xlabel("Height (m)")
axes[0].set_ylabel("Number of waterfalls")
axes[0].set_title("Histogram of Waterfall Height")

axes[1].hist(selected["WIDTH_IN_M"].dropna(), bins=15, edgecolor="black")
axes[1].set_xlabel("Width (m)")
axes[1].set_ylabel("Number of waterfalls")
axes[1].set_title("Histogram of Waterfall Width")
plt.tight_layout()
plt.show()

# 10. Features (inputs) and label (output)
model_data = selected.dropna(subset=["HEIGHT_IN_M", "WIDTH_IN_M", "OWNERSHIP"])
print(f"Rows used for modelling: {len(model_data)} of {len(selected)}")

X = model_data[["HEIGHT_IN_M", "WIDTH_IN_M"]]   # features
y = model_data["OWNERSHIP"]                     # label

# 11. Encode the categorical OWNERSHIP column as numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print("Encoding of OWNERSHIP as numbers:")
for code, label in enumerate(encoder.classes_):
    print(f"{label} -> {code}")

# 12. Train/test split with ratio 4:1 (80% train, 20% test)
try:
    # stratify keeps the class proportions similar in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded
    )
except ValueError:
    # falls back if a class has too few samples to stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2
    )
print(f"Train size = {len(X_train)}, Test size = {len(X_test)}")

# 13. Build the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# 14. Train the model on the training set
model.fit(X_train, y_train)
print("Model trained")

# 15. Predict the testing set ownership
y_pred = model.predict(X_test)
print("Predictions on test set:")
print("Predicted (decoded):", list(encoder.inverse_transform(y_pred)))
print("Actual (decoded):", list(encoder.inverse_transform(y_test)))

# 16. Evaluate: Accuracy, Confusion Matrix, Precision, Recall, F1-Score

# Binary problem (public/private): metrics are for the positive class (code 1).
# If there happen to be more than 2 classes, use the weighted average instead.
n_classes = len(encoder.classes_)
avg = "binary" if n_classes == 2 else "weighted"

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
recall = recall_score(y_test, y_pred, average=avg, zero_division=0)
f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

print("Evaluation:")
print(f"Accuracy : {accuracy:.4f}")
print("Confusion Matrix (rows = actual, columns = predicted):")
print(f"Class order: {list(encoder.classes_)}")
print(cm)
if avg == "binary":
    print(f"(Precision/Recall/F1 computed for positive class: '{encoder.classes_[1]}')")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
