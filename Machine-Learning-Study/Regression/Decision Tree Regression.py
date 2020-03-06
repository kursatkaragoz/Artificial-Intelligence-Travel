#%%  import Libraray and Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\decision-tree-regression-dataset.csv',sep=';',encoding="utf8",header=None)

# Data setimiz futbol sahasında seyircilerin sahaya uzaklık tipleri ve bilet fiyatlarını göstermektedir.
# Yukarıdan aşağıya uzaklık artmaktadır. 1. Tip en yakın koltuklar içindir. Bilet fiyatı yüksektir.

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%% decision tree regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)

#%% Visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="blue")
plt.xlabel("Tribun Level")
plt.ylabel("Price")
plt.show()