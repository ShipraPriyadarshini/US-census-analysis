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

x = allData['Working_Families_00']
#print(x)
y = allData['Response_Rate_00']
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


#plt.plot(range(0, 1))
#scale_factor = 0.2

#xmin, xmax = plt.xlim()
#ymin, ymax = plt.ylim()

#plt.xlim(xmin * scale_factor, xmax * scale_factor)
#plt.ylim(ymin * scale_factor, ymax * scale_factor) ""

#plt.legend(handles=[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])
