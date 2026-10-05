import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Admin\Desktop\NIT\1. NIT_Batches\2. EVENING BATCH\N_Batch -- 7.00PM -- JAN26\4. OCT\6th - poly\1.POLYNOMIAL REGRESSION\emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# linear regression visualizaton  
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

# again linear model build with 2nd degree
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#poly model
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() 

lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)

####### svr model 

from sklearn.svm import SVR
svr_model = SVR(kernel='poly', degree=4, gamma='auto',C = 10.0)
svr_model.fit(X, y)

svr_model_pred = svr_model.predict([[6.5]])
print(svr_model_pred)

####### knn regression model 

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='brute', p=2)
knn_model.fit(X,y) 

knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred)


####### decission tree model 
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X,y)

dt_model_pred = dt_model.predict([[6.5]])
print(dt_model_pred)


##### Random Forest 

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=23, random_state=0) 
rf_model.fit(X,y)

rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)













