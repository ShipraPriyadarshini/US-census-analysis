import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib.scale import get_scale_names
from matplotlib.axes import Axes, Subplot

#allData = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\All_data.csv', skiprows=0, delimiter=',')#print(y)
with open('All_data.csv', 'r') as f:
    allData = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\All_data.csv', skiprows=0, delimiter=',')

x = allData['Asian_10']
#print(x)
y = allData['Response_Rate_10']
#print(y)
x = np.array(x).reshape((-1, 1))
print(x)

y = np.array(y)
print(y)

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print("coefficient of determination:", r_sq)

print("intercept:", model.intercept_)
print("slope:", model.coef_)

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print("intercept:", new_model.intercept_)
print("slope:", new_model.coef_)

y_pred = model.predict(x)
print("predicted response:", y_pred, sep='\n')

slope = new_model.coef_
intercept = new_model.intercept_
line = slope*x+intercept

model = smf.ols('y ~ x', data=allData).fit()
print(model.summary())
print(model.pvalues)
