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

xtemp00 = ["Population_00","White_00","Black_00","Asian_00","Hispanic_00","Republican_00","Democratic_00","Independence_00","Conservative_00","Liberal_00","Green_00","Working_Families_00","AvgHousePrice_00"]

xtemp10 = ["Population_10","White_10","Black_10","Asian_10","Hispanic_10","Democratic_10","Republican_10","Independence_10","Conservative_10","Working_Families_10","Green_10","AvgHousePrice_10"]

xt = [xtemp00, xtemp10]

y00 = allData['Response_Rate_00']
y10 = allData['Response_Rate_10'] 


for nn in  xtemp00:
    x = allData[nn]
    y = y00
    #print (x)
    plt.figure(figsize=(10,10))
    sns.regplot(x, y)
    plt.plot([x],[y], 'o', label = x, color='red', markersize='3')#
    
    plt.title('Linear Regression Analysis: Response Rate & ' + nn )
      #  plt.xlabel(x, color='#1C2833')
       # plt.ylabel('Response Rate')
    plt.savefig( nn +'.pdf')
 #   plt.show()

    
for nn in  xtemp10:
    x = allData[nn]
    y = y10
    #print (x)
    plt.figure(figsize=(10,10))
    sns.regplot(x, y)
    plt.plot([x],[y], 'o', label = x, color='red', markersize='3')#
    
    plt.title('Linear Regression Analysis: Response Rate & ' + nn )
      #  plt.xlabel(x, color='#1C2833')
       # plt.ylabel('Response Rate')
    plt.savefig( nn +'.pdf')
 #   plt.show()

    

