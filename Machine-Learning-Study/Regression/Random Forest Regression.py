#%% INFORMATION

# =============================================================================
# Random Forest, Ensamble Learning'in bir üyesidir.
# Ensamble Learning: Aynı anda birden fazla algoritmayı kullanarak elde edilen bir modeldir.
# Aynı anda birden fazla machine learning algoritmasını kullanır sonuçların ortalamasını alır
# ve neticeye ulaşır. 
# Random Forest ise birden fazla ağaç yapısı ile çalışır. Çıkan sonuçların ortalaması alınarak
# random forest neticesi elde edilir.
# Random forest random ağaçlar ile çalışmaya başlamadan önce data içerisinden sub_data yani
# random olarak datanın belirli bir kısmı tahsis edilir.
# RandomForestRegressor() parametreleri
#   1.) n_estimators = Kullanacağı random ağaç sayısı
#   2.) random_state = data içerisinde seçilecek random sub_data kısmı
#    random olarak yapılır ise kod her run edilişte farklı sonuçlar verir.
#    Çünkü random sub_data her seferinde değişecektir. Bu değişmeyi engellemek için
#    random_state kullanılır.
#    random_state parameresine verilen değer bir ID niteliği taşır. Verdiğimiz bu id' ile programa
#    şu mantığı aşılarız: Eğer birdaha 42 id'ini görürsen yine aynı random bölümlemesini yap.

# =============================================================================
#%%  import Libraray and Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\random-forest-regression-dataset.csv',sep=';',encoding="utf8",header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators= 100, random_state=42)
rf.fit(x,y)

print("7.5 tribun seviyesinde bilet fiyatı nekadar :",rf.predict([[7.8]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

# Visualiza


plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="blue")
plt.xlabel("Tribun Level")
plt.ylabel("Ticket Price")
plt.show()












