# 0. importing necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
url = './auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

# 2. Visualize the data
print(dataset.describe())
dataset.info()
sns.pairplot(dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']])
plt.show()

# 3. Pre-process the data
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset[['Europe', 'Japan', 'USA']] = dataset[['Europe', 'Japan', 'USA']].astype(int)
print(dataset.head())
print(dataset.isna().sum())
dataset = dataset.dropna()
print(dataset.isna().sum())
X = dataset.drop(columns=['MPG'])
y = dataset['MPG']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50)
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X.columns, index=X_val.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

# 4. Train a Linear Regression model on all features
model = LinearRegression()
model.fit(X_train_s, y_train)

# 5. Score the model (R^2)
print(f"Train R^2:      {model.score(X_train_s, y_train):.4f}")
print(f"Validation R^2: {model.score(X_val_s, y_val):.4f}")
print(f"Test R^2:       {model.score(X_test_s, y_test):.4f}")

# 6. Single-feature models
single_features = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year']
single_scores = {}
for feat in single_features:
    xtr = X_train_s[feat].values.reshape(-1, 1)
    xva = X_val_s[feat].values.reshape(-1, 1)
    xte = X_test_s[feat].values.reshape(-1, 1)
    m = LinearRegression().fit(xtr, y_train)
    single_scores[feat] = {
        'train': m.score(xtr, y_train),
        'val': m.score(xva, y_val),
        'test': m.score(xte, y_test),
    }
    print(f"{feat:<14} train={single_scores[feat]['train']:.4f}  "
          f"val={single_scores[feat]['val']:.4f}  "
          f"test={single_scores[feat]['test']:.4f}")

# 7. Best 3 features
ranked = sorted(single_scores, key=lambda f: single_scores[f]['val'], reverse=True)
best3 = ranked[:3]
print("\nBest 3:", best3)

# 8. Combined score using the 3 best features
m3 = LinearRegression().fit(X_train_s[best3], y_train)
print(f"Features:       {best3}")
print(f"Train R^2:      {m3.score(X_train_s[best3], y_train):.4f}")
print(f"Validation R^2: {m3.score(X_val_s[best3], y_val):.4f}")
print(f"Test R^2:       {m3.score(X_test_s[best3], y_test):.4f}")

# 9. SelectKBest comparison
skb = SelectKBest(score_func=f_regression, k=3)
skb.fit(X_train_s[single_features], y_train)
skb_best3 = list(pd.Index(single_features)[skb.get_support()])
print("SelectKBest top 3:", skb_best3)
print("Manual top 3 (Q7):", best3)
mk = LinearRegression().fit(X_train_s[skb_best3], y_train)
print("SelectKBest model test R^2:", mk.score(X_test_s[skb_best3], y_test))
print("Manual model test R^2:     ", m3.score(X_test_s[best3], y_test))