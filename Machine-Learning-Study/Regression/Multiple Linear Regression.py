#%% INFORMATION
# Birden Fazla Coef'e sahip regression türüdür.
# b0,b1= kat sayılar (b0 = constant sabit, b1=coef değişen )
# x,y = featureler
# Simple Linear Regression =>  y = b0 + b1 * x
#                             maas = b0 + b1 * deneyim
# multipe Linear Regression => b0 + b1*x1 + b2*x2
#                             maas = b0 + b1*deneyim + b2*yas
#
# maas = dependent variable (bağımlı değişken deneyim ve yaş'a bağımlıdır)
# deneyim,yas = independent variable (bağımsız değişkenler)
# Amac = min(MSE)  , b0,b1,b2......?

#%%  import Libraray and Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\multiple-linear-regression-dataset.csv',sep=';',encoding="utf8")
x=df.iloc[:,[0,2]].values  # 0.feature and 2. feature all values
y=df.maas.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0:",multiple_linear_regression.intercept_)
print("b1,b2:",multiple_linear_regression.coef_)

#predict
multiple_linear_regression.predict(np.array([[10,35],[5,35]]))