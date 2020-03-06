#%% INFORMATION
# R2 Square: hata ölçüm yöntemidir.

# =============================================================================
#%%   R2 Square Metric forRandom Forest Regressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\random-forest-regression-dataset.csv',sep=';',encoding="utf8",header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators= 100, random_state=42)
rf.fit(x,y)

y_head = rf.predict(x)

from sklearn.metrics import r2_score

print("r_score: ",r2_score(y,y_head))

#%% Linear Regression for Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\linear-regression-dataset.csv',sep=';',encoding="utf8")
x= df.deneyim.values.reshape(-1,1)
y= df.maas.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x,y)
y_head = linear_reg.predict(x) # maas

from sklearn.metrics import r2_score

print("r_square: ", r2_score(y,y_head))




#%%














