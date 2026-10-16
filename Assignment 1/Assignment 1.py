# 4AI3 Assignment 1
# Nicolas Jao
# 400450151

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

'''
# Section 1
df = pd.read_csv("./titanic/train.csv")
print(df.head())
df_20 = df.iloc[:20].copy()
df.info()
mean = df_20["Age"].mean()
df_20["Age"] = df_20["Age"].fillna(mean)

# Section 2
age = df_20[["Age"]]
survival = df_20["Survived"]
coef = np.array([[0.02705495]])
bias = np.array([-0.75660013])
def log_odds(features, coefficients, bias):
    return np.dot(features, coefficients) + bias
result_log_odds = log_odds(age, coef, bias)
print(result_log_odds)
def sigmoid(log_odds):
    denominator = 1 + np.exp(-log_odds)
    return 1 / denominator
p_hat = sigmoid(result_log_odds)
def log_loss(probabilities, labels):
    return (-1 / labels.shape[0]) * np.sum(labels * np.log(probabilities) + (1-labels) * np.log(1-probabilities))
survival = survival.to_numpy().reshape(-1, 1)
cost = log_loss(p_hat, survival)
def predict(features, coefficients, bias, threshold):
    result_log_odds = log_odds(features, coefficients, bias)
    p_hat = sigmoid(result_log_odds)
    return np.where(p_hat >= threshold, 1, 0)
predicted_result = predict(age, coef, bias, 0.5)

# Section 3
logit_reg = LogisticRegression()
z_score_scaler = StandardScaler()
age_scaled = z_score_scaler.fit_transform(age)
logit_reg.fit(age_scaled, survival.ravel())
print(logit_reg.coef_)
print(logit_reg.intercept_)
result_log_odds_sk = log_odds(age_scaled, logit_reg.coef_, logit_reg.intercept_)
p_hat_sk = sigmoid(result_log_odds_sk)
cost_sk = log_loss(p_hat_sk, survival)
print(cost_sk)
df_20_new = df.iloc[20:40].copy()
df_20_new.info()
mean = df_20_new['Age'].mean()
df_20_new['Age'] = df_20_new['Age'].fillna(mean)
age_new = df_20_new[['Age']]
age_new_scaled = z_score_scaler.transform(age_new)
predictions = logit_reg.predict(age_new_scaled)
print(predictions)
print(df_20_new['Survived'].to_numpy())
'''

# Section 4
# 1
df = pd.read_csv("./titanic/train.csv")

# 2
df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})

# 3
df.info()
mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean)

# 4
df['UpperClass'] = df['Pclass'].apply(lambda x: 1 if x == 1 else 0)
df['MiddleClass'] = df['Pclass'].apply(lambda x: 1 if x == 2 else 0)
df['LowerClass'] = df['Pclass'].apply(lambda x: 1 if x == 3 else 0)

# 5
X = df[['Sex', 'Age', 'UpperClass', 'MiddleClass', 'LowerClass']]

# 6
y = df['Survived']

# 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# 8
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9
NicolasJaoModel = LogisticRegression()
NicolasJaoModel.fit(X_train_scaled, y_train)

# 10
score_test = NicolasJaoModel.score(X_test_scaled, y_test)
print(score_test)

# 11
Jack = np.array([0.0, 20.0, 0.0, 0.0, 1.0])
Rose = np.array([1.0, 17.0, 1.0, 0.0, 0.0])
Fabrizio = np.array([0.0, 22.0, 0.0, 0.0, 1.0])
NicolasJao = np.array([0.0, 22.0, 0.0, 1.0, 0.0])
fictional_passengers = np.array([Jack, Rose, Fabrizio, NicolasJao])
fictional_passengers_scaled = scaler.transform(fictional_passengers)

# 12
print(NicolasJaoModel.predict(fictional_passengers_scaled))

# When I ran it I got [0 1 0 0]. My model doesn't think I will survive... :(
# I think it's because I'm male, and most males didn't survive since they protected women and children