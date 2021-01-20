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
from statsmodels.api import add_constant

allData = pd.read_csv("All_data.csv")

#2000 Data
x = allData[['Population_00', 'White_00', 'Black_00', 'Asian_00', 'Hispanic_00', 'Republican_00',
'Democratic_00', 'Independence_00', 'Conservative_00', 'Liberal_00', 'Green_00', 'Working_Families_00', 'AvgHousePrice_00']]

#2010 Data
#x = allData[['Population_10', 'White_10', 'Black_10', 'Asian_10', 'Hispanic_10', 'Democratic_10',
 #'Republican_10', 'Independence_10', 'Conservative_10', 'Working_Families_10', 'Green_10', 'AvgHousePrice_10']]

#combined
#x = allData[['Population_00', 'White_00', 'Black_00', 'Asian_00', 'Hispanic_00', 'Republican_00',
#'Democratic_00', 'Independence_00', 'Conservative_00', 'Liberal_00', 'Green_00', 'Working_Families_00', 'AvgHousePrice_00',
#'Population_10', 'White_10', 'Black_10', 'Asian_10', 'Hispanic_10', 'Democratic_10','Republican_10',
#'Independence_10', 'Conservative_10', 'Working_Families_10', 'Green_10', 'AvgHousePrice_10']]
print(x)

y = allData['Response_Rate_00'] #2000
#y = allData['Response_Rate_10'] #2010
#y = allData[['Response_Rate_00', 'Response_Rate_10']] #combined
print(y)

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print("coefficient of determination:", r_sq)

new_model = LinearRegression().fit(x, y)
print("intercept:", new_model.intercept_)
print("slope:", new_model.coef_)

slope = new_model.coef_
intercept = new_model.intercept_
line = slope*x+intercept
model = smf.ols('y ~ x', data=allData).fit()
print(model.summary())
print(model.pvalues)

#plt.plot(x, y 'o', label = 'White', color='red', markersize='3')
plt.plot(x, y, 'o', markersize=3, color='red')
plt.plot(x, slope*x + intercept, color='#010d03')
plt.title('Linear Regression Analysis: Response Rate & Covariates 2000')
plt.xlabel('Covariates', color='#1C2833')
plt.ylabel('Response Rate')
plt.show()
