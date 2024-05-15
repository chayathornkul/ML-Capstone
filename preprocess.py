import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import numpy as np

df = pd.read_csv('data.csv')

df['Present Year'] = 2017
df['Age'] = df['Present Year'] - df['Year']
df.drop(['Present Year'], inplace = True, axis = 1)

df = df[df['Make'] != 'Bugatti']

# Reset the index if needed
df.reset_index(drop=True, inplace=True)


X = df.drop(['MSRP','Year','Number of Doors','Market Category', 'Popularity', 'Vehicle Style', 'Vehicle Size'], axis=1)

y = df['MSRP']


# Make,Model,Year,Engine Fuel Type,Engine HP,Engine Cylinders,Transmission Type,Driven_Wheels,Number of Doors,Market Category,Vehicle Size,Vehicle Style,highway MPG,city mpg,Popularity,MSRP

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

qual_cols = ['Make','Engine Fuel Type', 'Transmission Type', 'Driven_Wheels']
quan_cols = ['Age', 'Engine HP', 'Engine Cylinders','highway MPG','city mpg']

# Define the preprocessor
preprocessor = ColumnTransformer([
    ('encoder', OneHotEncoder(), qual_cols),
    ('scaler', MinMaxScaler(), quan_cols)
])

# Fit and transform the training data
X_train_prep = preprocessor.fit_transform(X_train)
X_train = pd.DataFrame(X_train_prep.toarray()).fillna(0)

X_test_prep = preprocessor.transform(X_test)
X_test = pd.DataFrame(X_test_prep.toarray()).fillna(0)

pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)






