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
allData['CombinedFactors10'] = allData['White_10'] + allData['Black_10'] + allData['Population_10'] + allData['Asian_10'] + allData['Hispanic_10'] + allData['Republican_10'] + allData['Democratic_10'] + allData['Independence_10'] + allData['Conservative_10'] + allData['Green_10'] + allData['Working_Families_10'] + allData['AvgHousePrice_10']

x = allData['CombinedFactors10']
y = allData['Response_Rate_10'] #y

plt.figure(figsize=(10,10))
sns.regplot(x, y)
plt.plot([x],[y], 'o', label = 'All factors', color='green', markersize='3')
plt.title('Linear Regression Analysis: Combined Factors for 2010')
plt.xlabel('Factors 2010', color='#1C2833')
plt.ylabel('Response Rate')
plt.show()
