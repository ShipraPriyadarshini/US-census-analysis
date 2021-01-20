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

allData = pd.read_csv('All_data.csv', skiprows=0, delimiter=',')

x = allData['Asian_10']
y = allData['Response_Rate_10'] #y

tempx =[]
plt.figure(figsize=(10,10))

sns.regplot(x, y)
plt.plot([x],[y], 'o', label = 'Asian', color='red', markersize='3')

plt.title('Linear Regression Analysis: Response Rate & Race:Asian for 2010')
plt.xlabel('Race:Asian', color='#1C2833')
plt.ylabel('Response Rate')
plt.show()

