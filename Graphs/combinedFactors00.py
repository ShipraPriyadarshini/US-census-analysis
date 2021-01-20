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

plt.figure(figsize=(10,10))
sns.regplot(x, y)
plt.plot([x],[y], 'o', label = 'All factors', color='green', markersize='3')
plt.title('Linear Regression Analysis: Combined Factors for 2000')
plt.xlabel('Factors 2000', color='#1C2833')
plt.ylabel('Response Rate')
plt.show()
