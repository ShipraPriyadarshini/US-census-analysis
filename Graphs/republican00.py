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

x = allData['Republican_00']
y = allData['Response_Rate_00'] #y

plt.figure(figsize=(10,10))

sns.regplot(x, y)
plt.plot([x],[y], 'o', label = 'Republican', color='pink', markersize='3')

plt.title('Linear Regression Analysis: Response Rate & Republican for 2000')
plt.xlabel('Republican', color='#1C2833')
plt.ylabel('Response Rate')
plt.show()

#
# clean code
