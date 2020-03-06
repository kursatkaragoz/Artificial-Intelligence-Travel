# =============================================================================
# # INFO Logistic Regression
# Logistic Regression, Classfication Algoritmalarındandır.
# Genelde 0 ya da 1 sonucu veren datalarda kullanılır.
# Yani Binary Classification içindir.
# Binary Classifcation iki farklı label'ı bulunan datasetlerdir.
# Logistic Regression en küçük neural network'dür.
# Simple Neual Network olarak tanımlanabilir.
# 
# # Computation Graph
# Matematiksel ifadeleri görselleştirmek için kullanılan bir yöntemdir.
# 
# Logistic Regressionlarda kullanılan matematiksel ifadeleri açıklamak için
# computation graphlar yardımcı olarak kullanılabilir.
# Dataset Bir Tümörün iyi huylu mu kötü huylu mu olduğu anlamak için tutulan verileri içeriyor.
# M=iyi huylu, B=kötü huylu ==> M=1, B=0
# Train etmek: ilgili ifadenin kendi modelimize uydurulmasıdır.
# =============================================================================
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('C:\\Users\\Kürşad\\Desktop\\Machine-Learning-Study\\datasets\\data.csv',sep=',',encoding="utf8")
# id and unnamed features delete
data.drop(["Unnamed: 32","id"],axis=1,inplace=True)
# Diagnosis features values is changed: M = 1 , B =0
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values
# independent variables, train dataset
x_data = data.drop(["diagnosis"],axis=1) 

# %% normalization dataset // all values in dataset are normalized and to astype numpy
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

# %% Train Test Split data==> 80% of data set for Train, 20% of data set for Test
# train_test_split method'unun içine x ve y eğitim için verildi.
# test_size=0.2 parametresi ilede veri setinin %20'sini test olarak ayrıldı.
# random_state parametresi ile "42" sayısını id olarak tut ve aynı işlem birdaha yapılırsa:
# aynı bölümleme işlemini yaparak aynı sonuçlara ulaşmamızı bize sağla.
# bu method sonucunda oluşacak resultlarıda belirtilen değişkenlere aktar.
# x'in %80 x_train , x'in %20'si x_test ; y'nin %80'i y_train, y'nin %20'si y_test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#columns and features place are changed
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

#print("x_train: ",x_train.shape)
#print("x_test: ",x_test.shape)
#print("y_train: ",y_train.shape)
#print("y_test: ",y_test.shape)

# %% parameter initialize and sigmoid function
# dimension parameter : features count ==> dimension=30

def initialize_weights_and_bias(dimension):
    #initialize
    w=np.full((dimension,1),0.01) # Create matrix array and all array values are 0.01
    b=0.0  #b0 = 0
    return w,b
# w,b = initialize_weights_and_bias(30)
def sigmoid(z):             #f(x) = 1 / (1 + (e ^ (-x)) ==> x=z
    y_head = 1 / (1 + np.exp(-z))   
    return y_head
# print(sigmoid(0))



# =============================================================================
# Her bir weightin, kendisine ait her bir piksel çiftleri ile çarpılması gerekir.
# x_train,y_train = pikseller, w=weightler ==> (30,1) * (30,455) matrixlerinin çarpması olamaz.
# 1. matrix'in sütunu ile 2. matrix'in satırı birbirine uymalıdır => (1,30) * (30,455) olmalıdır.
# Bu işlem sonucunda da (1,455) lik bir matrix elde edilir. ==> np.dot(w.T,x_train,y_train)
# Forward - Backward İşleminde yapılacaklar:
#    forward için:
#   1.) Pikseller ile ağırlıkları(weights) çarp ve bias ekle.
#   2.) y_head değerini sigmoid function ile hesapla.
#   3.) loss function formulünden yola çıkarak loss değerini hesapla
#   4.) cost functionu hesapla => sum(loss) / sample_count
#    
#    backward için:
#   1.)backward işleminde gerekli weighte göre türev al.
#   2.)backward işleminde gerekli bias'a göre türev al
#   cost ve gradients(derivative_weight, derivative_bias) return et.
    
# =============================================================================
    
def forward_backward_propagation(w,b,x_train,y_train): 
    # forward propagation
    z = np.dot(w.T,x_train) + b #  Z = b + (px1w1) + (px2w2) ... (pxnwn) (collection in image (+))
    y_head = sigmoid(z)         #  The z value entered the sigmoid. y_head value obtained
    loss = -y_train * np.log(y_head) -(1 - y_train) * np.log(1-y_head) # -(1 - y)log(1 -y_head) - ylogy_head
    cost = (np.sum(loss)) / x_train.shape[1]  # sum(loss) / sample count
    # Slope 1 finished 
    
    
    # backward propagation (weight ve bias'a göre türev almak için )
    # weight değerine göre türev alındı.
    # bias'a göre türev alındı
    # son olarak güncelleme kısmı gereklidir. Bu işlem update function'da yapılır.
    # 1/m * (y_head - y)
    deerivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1] #derivative based on weight
    derivative_bias = np.sum(y_head - y_train ) / x_train.shape[1] #derivative based on bias
    
    
    #weight and bias are derivates kept in dictionary(gradients) 
    # Güncel weight ve bias için slope(step) leri tutar. Yani weight ve bias'a görev türv sonuçları.
    # Güncellemede yapılacak işlem ==>   W:=w-step dir. ==> step = learning rate * derivative_weight(bias)
    gradients = {"derivative_weight": deerivative_weight, "derivative_bias":derivative_bias}
    return cost,gradients

# ==========================================INFO==================================
# Update işlemi arka arkaya forward propagation ve backward propagatin işleminin n defa yapılması işlemidir.
# Bu nedenle parametrelerimiz:
#     güncellenecek weightler             : w                    => parameter 1
#     güncellenecek biaslar               : b                    => parameter 2
#     forward için x_train input1         : x_train              => inputs features values
#     forward için y_train input_label    : y_train              => inputs class labels
#     slope için learning_rate değeri     : learning_rate        => hyper parameter 1
#     forward ve backward tekrar sayısı   : number_of_iteration  => hyper parameter 2
# Not: Gradients, weight ve bias'ın türevlerini tutar.
# costları tutarız çünkü nuber_of_itearation sayısını belirlemek için.
# learning_rate fazla veya az olursa öğrenme işlemi kazaya uğrayabilir.
# cost2'nin pek bir işlevi yoktur sadece her 10 adımda bir costları tutuyoruz 
# çünkü daha sonra rahatça göstermek için ve plot ettirmek için.
# =============================================================================
    
def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list=[]    # to store all cost values
    cost_list2=[]   # to store every 10 steps
    index=[]
    
    #updating (Learning) parameters is number_of_iteration times
    for i in range(number_of_iteration):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        # lets's update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        
        if i%10 ==0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i,cost))
        
    parameters = {"weight":w, "bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost")
    plt.show()
    
    return parameters, gradients, cost_list
# train completed    

# %% Predict Method
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    
    z=sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1])) #create prediction matrix and dimension = (1,sample_count)
    #if z is bigger than 0.5, our prediction is sign on (y_head = 1)
    #if z is smaller than 0.5, our prediction is sign zero (y_head=0)

    for i in range(z.shape[1]):
        if(z[0,i] <=0.5):
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
            
    return Y_prediction

# %% Test of Logistic Regression 

def logistic_regression(x_trai,y_train,x_test,y_test,learning_rate,num_iterations):
    #intialize
    dimension = x_train.shape[0] #feature count is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    
    #update method for forward and backward propagation
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    
    y_prediction_test = predict(parameters["weight"], parameters["bias"],x_test)
    #y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)
    
    #Accuarcy,print test error  (Accuracy başaranı oranını verir)
    #print("Train Accuarcy: {}%".format(100- np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("Test Accucarcy: {}%".format(100 - np.mean(np.abs(y_prediction_test - y_test))*100))


logistic_regression(x_train,y_train,x_test,y_test,learning_rate=3,num_iterations=1665)

#%% Logistic Regression with Sklearn Library
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("Test Accuracy : {}".format(lr.score(x_test.T,y_test.T)))
# %%
from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))


