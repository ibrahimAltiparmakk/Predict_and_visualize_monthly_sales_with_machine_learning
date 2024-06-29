# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 14:30:54 2024

@author: sarib
"""

import pandas as pd  #veri işlemlerinde kullanılır

import numpy as np  #büyük sayılar ya da hesaplama işlemleri için kullandığımız kütüphane

#kodlar

#-------------------- veri yükleme ----------------

#pandas kütüphanesini verileri okumak için kullanıyoruz
#veri dosyamız csv dosyası olduğu için read_csv diyoruz
veriler = pd.read_csv("satislar.csv")
print(veriler)

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
# scaler = ölçekleyici
# standart hale getirme
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

# x kısmı bağımsız (aylar) , y kısmı bağımlı (satışlar)
# X_train ve Y_train bilgilerini alarak bir model inşa et diyoruz
# traindeki verilerden eğiticeğiz , test de tahin yapacağız


#------------------ model inşası (lineer regression)--------------------
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
#lr.fit(X_train, Y_train) komutu, LinearRegression modelinin eğitim verileriyle (X_train ve Y_train) eğitilmesini sağlar.

tahmin = lr.predict(x_test) #predict = tahmin
#modelimizi eğittikten sonra yukarıdaki kısım testteki bağımsız değişken verdikten sonra bağımlı değikeni (Y_test) tahmin etmesini istiyoruz 
#tahmindeki değerler tahmin edilmiş değerlerdir (Y_test ile karşılaştırabilirsiniz)

x_train = x_train.sort_index()#index e göre sıralama
y_train = y_train.sort_index()

import matplotlib.pyplot as  plt #çizimler için kullanılır

plt.plot(x_train,y_train) #grafik çizdirme
plt.plot(x_test,lr.predict(x_test)) # tahmini grafiğe dökme 
#yukarıdaki kısımda ilk bölüme x_test i verdik , 2. bölüme de tahmin edilen verileri(lr.predict(x_test)) verdik
plt.title("Aylara Göre Satış") # Grafiğe başlık verebilirsin
plt.xlabel("Aylar") # x ekseninin başlığı
plt.ylabel("Satışlar") # y ekseninin başlığı


#grafikteki düz çizgi tahmin edilen kısımlardır
