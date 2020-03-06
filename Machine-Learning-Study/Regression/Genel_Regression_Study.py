# =============================================================================
# INFORMATION

# 1.) Why Linear Regression bölümünü incelediğimizde, Scattera bakarsak değerlerin Linear bir şekilde
# arttığı görülmektedir. Bir Regresyon uygulayacak olursak mantık olarak line'ımızın linear bir
# line olması gerekir.
# Bu nedenle  Linear Regression yöntemi ile tahmin işlemi gerçekleştirdik.
# 
# =============================================================================
#%% Data Read
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\column_2C_weka.csv',sep=',',encoding="utf8")
#data filter
df = df[df['class'] == "Abnormal"]

#%% 1.) Why LinearRegression ?
#x = df.loc[:,'pelvic_incidence'].values.reshape(-1,1)
#y=df.loc[:,'sacral_slope'].values.reshape(-1,1)
x = np.array(df.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(df.loc[:,'sacral_slope']).reshape(-1,1)


# 2.) Linear Regression 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# do Fit
lr.fit(x,y)
# predict space
x_ = np.linspace(min(x),max(x),num=210).reshape(-1,1) #Eşit aralıklarla başlangıç ile bitiş arası sayı üret
# Predict
y_head = lr.predict(x_)
print("R^2 Score :,",lr.score(x,y))

plt.plot(x_,y_head,color="red",linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('Pelvic Incidence')
plt.ylabel('Sacral Slope')
plt.show()


#%%
x=df.loc[:,'pelvic_incidence'].values.reshape(-1,1)
y = np.array(df.loc[:,'sacral_slope']).reshape(-1,1)

# do polynomial fit
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 2) # 2. derece kalıp oluştur
x_polynomial = polynomial_regression.fit_transform(x)  # kalıba x'i sok fit et aktar

# polynomial fit is do linear fit
from sklearn.linear_model import LinearRegression
lr2 = LinearRegression()
lr2.fit(x_polynomial,y)
print("R^2 Score :,",lr2.score(x_polynomial,y))

y_head = lr2.predict(x_polynomial)
plt.scatter(x,y,color="red")
plt.plot(x,y_head,color="black")
plt.show()

# =============================================================================
#%% DATA Read for Polynomial Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\corona.csv',sep=',',encoding="utf8",header=None)
df = df.loc[:,[2,3,7,9,11]]
# =============================================================================

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,4].values.reshape(-1,1)

#%%
# transform polynomial regression fit
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree=2)
x_polynomial = polynomial_regression.fit_transform(x)

# linear fit
from sklearn.linear_model import LinearRegression
l_regression = LinearRegression()
l_regression.fit(x_polynomial,y)
y_head = l_regression.predict(x_polynomial)
print("R^2 Score :",l_regression.score(x_polynomial,y))

plt.scatter(x,y,color="black",marker=r'$\clubsuit$')
plt.plot(x,y_head,color="red")
plt.show()

#%% Data Read for Multiple Regression
# "2" Variable'nin "3" ve "7". variabe'e bağımlı olduğunu varsayalım.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\corona.csv',sep=',',encoding="utf8",header=None)
df = df.loc[:,[2,3,7,9,11]]

from sklearn.linear_model import LinearRegression
#x=df.loc[:,[3,7]].values
x=df.iloc[:,[1,2]].values # 1. index (3) and 2. index (7) feature's values
y=df[2].values.reshape(-1,1)
multiple_lr=LinearRegression()
multiple_lr.fit(x,y)

print("b0", multiple_lr.intercept_)
print("b1,b2", multiple_lr.coef_)
y_head = multiple_lr.predict(x)

#%% Data Read for DecisionTreeRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\corona.csv',sep=',',encoding="utf8",header=None)
df = df.loc[:,[2,3,7,9,11]]

x = df.iloc[:,3].values.reshape(-1,1)
y = df.iloc[:,4].values.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

y_head = tree_reg.predict(x)

#Visualiza
plt.scatter(x,y,color="blue")
plt.plot(x,y_head,color="red")
plt.xlabel(" Feature 2")
plt.ylabel(" Feature 11")
plt.show()

# r2

from sklearn.metrics import r2_score
print("R^2 Score: ",r2_score(y,y_head))
print("R^2 Score: ",tree_reg.score(x,y))


#%% Random Forest Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\corona.csv',sep=',',encoding="utf8",header=None)
df = df.loc[:,[2,3,7,9,11]]

x = df.iloc[:,3].values.reshape(-1,1)
y = df.iloc[:,4].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100,random_state=23)
rf.fit(x,y)
y_head = rf.predict(x)

#Visualiza
plt.scatter(x,y,color="blue")
plt.plot(x,y_head,color="red")
plt.xlabel(" Feature 2")
plt.ylabel(" Feature 11")
plt.show()

# error score
from sklearn.metrics import r2_score
print("R^2 Score: ",r2_score(y,y_head))
print("R^2 Score: ",rf.score(x,y))








