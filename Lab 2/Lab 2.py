# 1. import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 2. Upload and Preview the Dataset
DATA_PATH = "honeyproduction.csv"
df = pd.read_csv(DATA_PATH)
print("First 5 rows of the dataset:")
print(df.head())

# 3. Inspect the Dataset
print("Dataset info:")
print(df.info())

# 4. Calculate the Average Price Per Year
avg_price_per_year = df.groupby("year")["priceperlb"].mean().reset_index()
print("Average price per pound by year:")
print(avg_price_per_year)

# 5. Prepare Feature and Target Variables
X = avg_price_per_year["year"].values.astype(float)         # feature: Year
y = avg_price_per_year["priceperlb"].values.astype(float)   # target: priceperlb

# 6. Visualize the Data
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Actual data")
plt.xlabel("Year")
plt.ylabel("Average Price per Pound ($)")
plt.title("Average Honey Price per Pound by Year")
plt.legend()
plt.tight_layout()
plt.savefig("scatter_year_vs_price.png")
plt.show()

# 7 & 8. Build and Train Linear Regression Models

# Model 1: Linear Regression from scratch
X_mean, X_std = X.mean(), X.std()
X_scaled = (X - X_mean) / X_std
def train_linear_regression_scratch(X, y, lr=0.01, epochs=10000):
    # Train y = w*X + b using batch gradient descent
    n = len(X)
    w, b = 0.0, 0.0
    for i in range(epochs):
        y_pred = w * X + b
        error = y_pred - y
        dw = (2 / n) * np.sum(error * X)
        db = (2 / n) * np.sum(error)
        w -= lr * dw
        b -= lr * db
    return w, b
w_scaled, b_scaled = train_linear_regression_scratch(X_scaled, y, lr=0.01, epochs=10000)
# Convert weights back from the standardized scale to the original Year scale:
# y = w_scaled * ((X - X_mean)/X_std) + b_scaled
#   = (w_scaled / X_std) * X + (b_scaled - w_scaled * X_mean / X_std)
w_scratch = w_scaled / X_std
b_scratch = b_scaled - (w_scaled * X_mean / X_std)

# Model 2: scikit-learn's LinearRegression
X_sklearn = X.reshape(-1, 1)
sklearn_model = LinearRegression()
sklearn_model.fit(X_sklearn, y)

# 9. Obtain the Trained Weights
w_sklearn = sklearn_model.coef_[0]
b_sklearn = sklearn_model.intercept_
print("--- Trained Weights ---")
print(f"Model 1 (Scratch)  -> slope (w): {w_scratch:.6f}, intercept (b): {b_scratch:.6f}")
print(f"Model 2 (sklearn)  -> slope (w): {w_sklearn:.6f}, intercept (b): {b_sklearn:.6f}")

# 10. Make Predictions
y_pred_scratch = w_scratch * X + b_scratch
y_pred_sklearn = sklearn_model.predict(X_sklearn)

# 11. Visualize Predicted and True Values
plt.figure(figsize=(9, 6))
plt.scatter(X, y, color="black", marker="o", label="True values")
plt.plot(X, y_pred_scratch, color="red", linestyle="--", marker="x", label="Model 1: Scratch prediction")
plt.plot(X, y_pred_sklearn, color="green", linestyle="-", marker="^", label="Model 2: sklearn prediction")
plt.xlabel("Year")
plt.ylabel("Average Price per Pound ($)")
plt.title("True vs Predicted Honey Price per Pound")
plt.legend()
plt.tight_layout()
plt.savefig("predictions_comparison.png")
plt.show()

# 12. Examine the Hypothesis Lines
print("--- Hypothesis / Line Equations ---")
print(f"Model 1 (Scratch):  price = {w_scratch:.6f} * Year + ({b_scratch:.6f})")
print(f"Model 2 (sklearn):  price = {w_sklearn:.6f} * Year + ({b_sklearn:.6f})")

# 13. Make Future Predictions
future_year = 2024
pred_scratch_2024 = w_scratch * future_year + b_scratch
pred_sklearn_2024 = sklearn_model.predict(np.array([[future_year]]))[0]
print(f"--- Predictions for Year {future_year} ---")
print(f"Model 1 (Scratch)  predicted price per lb: ${pred_scratch_2024:.4f}")
print(f"Model 2 (sklearn)  predicted price per lb: ${pred_sklearn_2024:.4f}")

# 14. Compare the Results
mse_scratch = np.mean((y - y_pred_scratch) ** 2)
mse_sklearn = np.mean((y - y_pred_sklearn) ** 2)
print("--- Model Comparison ---")
print(f"MSE (Model 1 - Scratch): {mse_scratch:.6f}")
print(f"MSE (Model 2 - sklearn): {mse_sklearn:.6f}")
print(f"Slope difference: {abs(w_scratch - w_sklearn):.8f}")
print(f"Intercept difference: {abs(b_scratch - b_sklearn):.8f}")
print(f"2024 prediction difference: ${abs(pred_scratch_2024 - pred_sklearn_2024):.6f}")

print("""
Discussion:
- Learned weights: The slope and intercept from the scratch model (trained via
  gradient descent on standardized Year values, then converted back) should be
  very close to sklearn's closed-form (least squares) solution. Any small
  differences come from gradient descent's iterative approximation, learning
  rate, and number of epochs, rather than a fundamentally different model. From
  the few times I've run the program, there was no difference between them here
  (difference of nearly 0).
- Predictions: Because the weights are nearly identical, predictions on the
  training data and the 2024 forecast should also be very close between the
  two models, typically differing by a very small fraction of a cent per lb.
  From the few times I've run the program, there was no difference between them
  here (difference of nearly 0).
- Accuracy/observations: sklearn's LinearRegression solves for the optimal
  weights directly (ordinary least squares), so it is exact and fast for this
  small dataset. The "from scratch" implementation approximates the same optimum
  through iterative gradient descent, and its accuracy depends on choosing an
  appropriate learning rate and enough training epochs to converge. With
  proper tuning (as done here via feature standardization), both models reach
  essentially the same MSE and hypothesis line.
""")
