#%%  import Libraray and Dataset
import pandas as pd
# %matplotlib inline # Show matplotlib's output
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\linear-regression-dataset.csv',sep=';',encoding="utf8")

# Array To DataFrame (Array=linear-regression-dataset)
# df= pdf.DataFrame(linear-regression-dataset,columns=["deneyim","maas"])

#%% Plot Data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("Deneyim")
plt.ylabel("Maas")
plt.show()

#%% Information
# b0,b1= kat sayılar (b0 = constant sabit, b1=coef değişen )
# x,y = featureler
# Bir Line'ı fit etmek için kullanılan yöntem MSE'dir.
# y=> Noktanın Kendisi
# y_head=> Çizilen Doğrudaki y'e en yakın doğru ( Line Üzerindeki Noktalar)
# Residual = y- y_head  => residual noktamız ile doğruya karşılık nokta arası farkdır.
# Fark (-) çıkabileceği için errorleri kaybetmemek için residual'un karesi alınır.
# Bir nokta için bu işlem yapılır. Bu işlemin bütün noktalara uygulanıp çıkan sonuçların
# toplanmasıyla topla error elde edilir
# MSE = sum(residual^2)/n ile sonuç elde edilir. n=sample sayısı
# MSE = 1/n E(yi - y_head_i)^2 diğer bir değişle böyledir. 

#%% sklearn library
from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg = LinearRegression() # create linear regression model
x=df.deneyim.values.reshape(-1,1)   # pandas.core.series to numpy, (.values=pandas=>numpy) (.reshape = 14, => 14,1)
y=df.maas.values.reshape(-1,1)     
linear_reg.fit(x,y)     # do fit

#%% prediction
# Y= B0 + B1*X
import numpy as np
b0=linear_reg.predict([[0]])   # x=0 => b0 = ?))
print('b0',b0)
b0_=linear_reg.intercept_    # Bu method ile b0 bulunur.
print('b0',b0_)
b1_=linear_reg.coef_  # Bu method is b1'i verir.
print('b1',b1_)

# maas = b0 + b1*deneyim
# maas = 1663 + 1138*deneyim
maas_yeni = 1663 + 1138*11
print(maas_yeni)  # yöntem 1  x=11 => b0 = ?
print(linear_reg.predict([[11]])) # yöntem2  x=11 => b0 ?

#%% Visualize Line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) # deneyim
plt.scatter(x,y)
y_head = linear_reg.predict(array)   # do predict maas
plt.plot(array,y_head,color="red")
plt.show()
