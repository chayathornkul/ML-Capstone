import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np



# import preprocessed data

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')


# Convert the DataFrame to a CSR matrix
X_train_prep = csr_matrix(X_train.values)
X_test_prep = csr_matrix(X_test.values)


# Train and evaluate Linear Regression
lr = LinearRegression()
lr.fit(X_train_prep, y_train)
y_pred_lr = lr.predict(X_test_prep)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression Mean Squared Error: {mse_lr}")



plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, color='skyblue')
plt.xlabel('Actual MSRP')
plt.ylabel('Predicted MSRP')
a, b = np.polyfit(y_test.values.flatten(),y_pred_lr.flatten(), 1)
plt.plot(y_test, a*y_test+b, c='red')
plt.text(3, 6, f"The slope is {a:.2f}", fontsize=12, color='red')
plt.title('Actual MSRP vs predicted MSRP from Linear Regression')
plt.show()




# Train and evaluate K-Nearest Neighbors
knn = KNeighborsRegressor()
knn.fit(X_train_prep, y_train)
y_pred_knn = knn.predict(X_test_prep)
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f"K-Nearest Neighbors Mean Squared Error: {mse_knn}")


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_knn, color='skyblue')
plt.xlabel('Actual MSRP')
plt.ylabel('Predicted MSRP')
a, b = np.polyfit(y_test.values.flatten(),y_pred_knn.flatten(), 1)
plt.plot(y_test, a*y_test+b, c='red')
plt.text(3, 6, f"The slope is {a:.2f}", fontsize=12, color='red')
plt.title('Actual MSRP vs predicted MSRP from K-Nearest Neighbors Regressor')
plt.show()



# Train and evaluate Decision Tree Regressor
dt = DecisionTreeRegressor()
dt.fit(X_train_prep, y_train)
y_pred_dt = dt.predict(X_test_prep)
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f"Decision Tree Regressor Mean Squared Error: {mse_dt}")


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_dt, color='skyblue')
plt.xlabel('Actual MSRP')
plt.ylabel('Predicted MSRP')
a, b = np.polyfit(y_test.values.flatten(),y_pred_dt.flatten(), 1)
plt.plot(y_test, a*y_test+b, c='red')
plt.text(3, 6, f"The slope is {a:.2f}", fontsize=12, color='red')
plt.title('Actual MSRP vs predicted MSRP from Decision Tree Regressor')
plt.show()


# plot the MSE of each model

mse_values = [mse_lr, mse_knn, mse_dt]
models = ['Linear Regression', 'KNN', 'Decision Tree']

plt.figure(figsize=(8, 6))
plt.bar(models, mse_values, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.title('Comparison of MSE Values for Different Models')
plt.show()

