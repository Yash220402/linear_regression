import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# linear regression on the training set
LR = LinearRegression()
LR.fit(X_train, y_train)

# making predictions
y_pred = LR.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, LR.predict(X_train), color='red')

# Visualising the Test set results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, LR.predict(X_train), color='red')
# plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
