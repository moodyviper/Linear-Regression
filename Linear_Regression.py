

# Assignment on Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

data = {'Age':[18,19,20,21,22,23,24,25,26,27,28],'Height':[76.1,77,78.1,78.2,78.8,79.9,81.1,81.2,81.8,82.8,83.5] }
print(data)

data = pd.DataFrame(data,columns=['Age','Height'])
print(data)

X = data.iloc[:,:-1].values
y = data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.coef_)

print(regressor.intercept_)

y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

#scatter plot
plt.scatter(data.Age,data.Height,c='b',marker='*')
plt.xlabel('Age',c='g',fontsize=14)
plt.ylabel('Height',c='g',fontsize=14)
plt.title('Scatter Plot of Age Vs height',c='r',fontsize=15)
