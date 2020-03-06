#%%
# b0,b1= kat sayılar (b0 = constant sabit, b1=coef değişen )
# x,y = featureler

# linear regression => y= b0 + b1*x
# multiple regression => y= b0 + b1*x1 + b2*x2 ...
# Polynomial Linear Regression
# Linear olmayan datalar için kullanılır. Parbolic linelar için kullanılır.
# Bu Dataset belirli bir değerden sonra artışını durduran bir datasettir.
# y olayı polynomial'de değişkenlik gösterir. Örneğin => y= b0 + b1*x + b2*x^2 ... bn*x^n


#%%  import Libraray and Dataset
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\polynomial-regression.csv',sep=';',encoding="utf8")

#%%
x=df.araba_fiyat.values.reshape(-1,1)
y=df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("Araba_Fiyat")
plt.ylabel("Araba_max_hiz")
#plt.show()
#%% Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)

#%%predict
y_head = lr.predict(x)
plt.plot(x,y_head,color="red",label="linear")
#plt.show()
print("10 bin tllik arabanın tahmini hızı :",lr.predict([[10000]])) # Linear Regression yaklaşımı kötü sonuç verir.

#%% Polynomial Regression y=b0 + b1*x1 + b2*x^2
# Amaç 2. dereceden polynomial oluştur. (degree=2 ==> 2.dereceye kadar işlem yap (x^2))
# 2. dereceden fit işlemi gerçekleştir ve tut.
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 2) #2. derece regresyon kalıbı oluştur
x_polynomial = polynomial_regression.fit_transform(x)  # kalıba x'i  sok, fit et ve aktar

#%% fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y) 

#%%
y_head2 = linear_regression2.predict(x_polynomial)
plt.plot(x,y_head2,color="green",label="Polynomial2")
plt.legend()
#plt.show()
#%% Degree ile oynayarak MSE minimuma düşürülebilir, farklı sonuçlar alınabilir
# Degree = 4 iken sonuca bakalım
#%%
polynomial_regression = PolynomialFeatures(degree = 4) 
x_polynomial = polynomial_regression.fit_transform(x) 

#%% fit2
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y) 

#%%
y_head2 = linear_regression2.predict(x_polynomial)
plt.plot(x,y_head2,color="black",label="Polynomial4")
plt.legend()
plt.show()