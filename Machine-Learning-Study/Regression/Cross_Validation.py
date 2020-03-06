
#%% =============================================================================
# INFO: The technique OF CROSS VALIDATION is K FOLD Cross Validatiıon
# 
# Çapraz Doğrulama (Cross Validation): Bir hata bulma, Bir amac doğrultusunda oluşturulmuş
# machine learning modelinin başarısının sınanması için kullanılan bir yöntemdir.
# Çalışma Mantığı:
#     Veri kümesini; eğitim veri kümesi ve test kümesi olmak üzere iki kısma ayırır.
#     Ayırma işlemi çeşitli şekillerde yapılabilir, oranlar değişiklik gösterebilir.
#     Bu oranlara göre veri seti içerisinden rastgele olarak eğitim ve test verileri oluşturulur.
#     random_state parametresi ile bu rastgele veri atama işlemi durdulabilir.
#     Eğitim için ve test için sabit veriler kullanılabilir. Böylelikle sabit resultlar elde edilir.
#     
#     K Fold cross validation Cross Validation yöntemlerinden bir tanesidir.
#     Veri setini k adet eşit parçaya böler. Bu eşit parçaların bir kısmını eğitim
#     bir kısmını test olarak kullanır.
#     Eğitim için ayrılan veriler ile model eğitilir ve test verileri ile test edilir.
#     Bir sonraki adımlarda test ve eğitim verileri yer değiştirerek kombinasyon tamamlanır.
#     Çıkan error resultların rate'i hesaplanır. Ortalaması alındığı için veri setini
#     ayırma işleminin nereden başladığının bir önemi yoktur.
# =============================================================================

#%% Data import 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\column_2C_weka.csv',sep=',',encoding="utf8")

#%%
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

x = np.array(df.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(df.loc[:,'sacral_slope']).reshape(-1,1)

x_ = np.linspace(min(x),max(x)).reshape(-1,1)
reg.fit(x,y)
y_predict = reg.predict(x_)

print("R^2 Score: ",reg.score(x,y))

plt.scatter(x,y,color="blue")
plt.plot(x_,y_predict,color="black")
plt.show()

# CROSS VALIDATION => K FOLD CROSS VALIDATION
from sklearn.model_selection import cross_val_score
k=5
cv_result = cross_val_score(reg,x,y,cv=k)
print('CV Scores: ',cv_result)
print('CV scores average: ',np.sum(cv_result)/k)