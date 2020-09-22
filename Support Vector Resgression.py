#importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing data 

dataset = pd.read_csv('C:/Users/My Pc/Desktop/machine learning tests/regression/Polynomial Regression/Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2:3].values

""" # splitting data into trainingset and testset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( dp, indp, test_size= 0.2, random_state= 0)"""

# Feature scalling  
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


# Fitting the Regression to the dataset
from sklearn.svm import SVR
regresor = SVR(kernel = 'rbf')
regresor.fit(x, y)

# predicitng a new result with polynomial regresion
y_pred = sc_y.inverse_transform(regresor.predict(sc_x.transform(np.array([[6.5]]).reshape(1, 1)))) 

# visualising SVR regression
plt.scatter(x, y, c = 'b')
plt.plot(x, regresor.predict(x), c = 'r')
plt.title('salary vs possiotion (SVR)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()


# visualising svr regression
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, c = 'g')
plt.plot(x_grid, regresor.predict(x_grid), c = 'r')
plt.title('salary vs possiotion (SVR)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

