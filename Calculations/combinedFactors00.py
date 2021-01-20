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
from scipy.stats import linregress
import seaborn as sns

allData = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\All_data.csv', skiprows=0, delimiter=',')
allData['CombinedFactors00'] = allData['White_00'] + allData['Black_00'] + allData['Population_00'] + allData['Asian_00'] + allData['Hispanic_00'] + allData['Republican_00'] + allData['Democratic_00'] + allData['Independence_00'] + allData['Conservative_00'] + allData['Green_00'] + allData['Working_Families_00'] + allData['AvgHousePrice_00']

x = allData['CombinedFactors00']
y = allData['Response_Rate_00'] #y

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
