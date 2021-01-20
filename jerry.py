import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#import statsmodels.formula.api as smf

allData1 = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\All_data.csv', skiprows=0)

resp10 = resp10 = allData1["Response_Rate_10"]
resp00 = allData1["Response_Rate_00"]
resp = resp10 - resp00

pop10 = allData1["Population_10"]
pop00 = allData1["Population_00"]
pop = pop10 - pop00

white10 = allData1["White_10"]
white00 = allData1["White_00"]
white = white10 - white00

black10=allData1["Black_10"]
black00=allData1["Black_00"]
black = black10-black00

hispan10=allData1["Hispanic_10"]
hispan00=allData1["Hispanic_00"]
hispan = hispan10 - hispan00

asian10=allData1["Asian_10"]
asian00=allData1["Asian_00"]
asian = asian10 - asian00

repub10=allData1["Republican_10"]
repub00=allData1["Republican_00"]
repub = repub10-repub00

demo10=allData1["Democratic_10"]
demo00=allData1["Democratic_00"]
demo = demo10-demo00

indep10 = allData1["Independence_00"]
indep00 = allData1["Independence_10"]
indep = indep10 - indep00
print(indep00)
conserv10 = allData1["Conservative_10"]
conserv00 = allData1["Conservative_00"]
conserv = conserv10 - conserv00

#PROBLEM
ahp10=allData1["AvgHousePrice_10"]
ahp00=allData1["AvgHousePrice_00"]
ahp=ahp10-ahp00

#PROBLEM
wf10=allData1["Working_Families_10"]
wf00=allData1["Working_Families_00"]
wf=wf10-wf00

y10 = resp10/pop10
y00 = resp00/pop00
diffR = y10-y00
print(diffR)
