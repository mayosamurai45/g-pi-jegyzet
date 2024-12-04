# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:18:33 2023

@author: martin
"""
#%%
#Imports
import numpy as np;  # importing numerical library
from matplotlib import pyplot as plt;  # importing the MATLAB-like plotting tool
from sklearn.datasets import load_digits; # importing digit dataset
from sklearn.model_selection import train_test_split; # importing splitting
from sklearn.decomposition import PCA;  # importing PCA

import seaborn as sns;  # importing the Seaborn library

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import ConfusionMatrixDisplay; #  importing performance metrics


from sklearn.cluster import KMeans; # Class for K-means clustering
from sklearn.metrics import davies_bouldin_score;  # function for Davies-Bouldin goodness-of-fit 

#%%
#1. Olvassa be a digits beépített adatállományt és írassa ki a legfontosabb jellemzőit (rekordok száma, attribútumok száma és osztályok száma). (3 pont)
digits = load_digits();
rekordok_szama = digits.data.shape[0]; # number of records
attributumok_szama = digits.data.shape[1]; # number of attributes
osztalyok_szama=digits.target_names.shape[0]

print(f"Rekordok szama: {rekordok_szama}\nAttributumok szama: {attributumok_szama}\nOsztalyok szama: {osztalyok_szama}")


#%%
#2. Készítsen többdimenziós vizualizációt a mátrix ábra segítségével (pairplot). (4 pont)
digits_df = load_digits(as_frame=True);
sns.set(style="ticks");
#sns.pairplot(digits_df.frame, hue="target");


#%%
#3. Particionálja az adatállományt 80% tanító és 20% tesztállományra. Keverje össze a rekordokat és a véletlenszám-generátort inicializálja az idei évvel. (3 pont)

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, shuffle=True, test_size=0.2, random_state=2023);


#%%
#4. Végezzen felügyelt tanítást az alábbi modellekkel és beállításokkal: 
    #döntési fa (4 mélység, entrópia homogenitási kritérium), 
    #logisztikus regresszió (liblinear solverrel) és 
    #neurális háló (1 rejtett réteg 4 neuronnal, logisztikus aktivációs függvény). A teszt score alapján hasonlítsa össze az illesztett modelleket, melyeket nyomtasson ki. (10 pont)

#Döntési fa
class_tree = DecisionTreeClassifier(criterion='entropy',max_depth=4);

class_tree.fit(X_train, y_train);
score_train_tree = class_tree.score(X_train, y_train); # Goodness of tree on training dataset
score_test_tree = class_tree.score(X_test, y_test); # Goodness of tree on test dataset

#Logisztikus regresszió
reg = LogisticRegression(solver='liblinear')

reg.fit(X_train, y_train);
score_train_logreg = reg.score(X_train, y_train)
score_test_logreg = reg.score(X_test, y_test);

#Neurális Háló
neural_classifier = MLPClassifier(hidden_layer_sizes=(4),activation='logistic',max_iter=10000);  #  number of hidden neurons: 4

neural_classifier.fit(X_train,y_train);
score_train_neural = neural_classifier.score(X_train,y_train);  #  goodness of fit
score_test_neural = neural_classifier.score(X_test,y_test);  #  goodness of fit
ypred_neural = neural_classifier.predict(X_test);   # spam prediction

print(f"Dontesi fa pontszama: {score_test_tree}")
print(f"Logisztikus regresszio pontszama: {score_test_logreg}")
print(f"Neuralis halo pontszama: {score_test_neural}")

#%%
#5-6. Számolja ki és ábrázolja az 4. pont legjobb modelljére a teszt tévesztési mátrixot. (4 pont)
disp = ConfusionMatrixDisplay.from_predictions(y_test, reg.predict(X_test), display_labels=digits.target_names.tolist())
plt.show()

#%%
#7. Végezzen nemfelügyelt tanítást a K-közép módszerrel az input attribútumokon. 
    #Határozza meg az optimális klaszterszámot 30-ig a DB indexszel. 
    #Az optimális klaszterszám mellett vizualizálja a klasztereket egy pontdiagrammon, 
    #ahol a két koordináta egy 2 dimenziós PCA eredménye. (13 pont)

# Finding optimal cluster number
Max_K = 30;  # maximum cluster number
SSE = np.zeros((Max_K-2));  #  array for sum of squares errors
DB = np.zeros((Max_K-2));  # array for Davies Bouldin indeces
n_c=2;
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2023);
    kmeans.fit(X_test);
    cancer_labels = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(X_test,cancer_labels);

fig = plt.figure(4);
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();

nr_of_clusters,=np.where(DB==DB.max())
print(f"Biggest score at {nr_of_clusters} clusters: {DB.max()}")


kmeans = KMeans(n_clusters=(nr_of_clusters[0]+2), random_state=2023);  # instance of KMeans class
kmeans.fit(digits.data);   #  fitting the model to data
digits_labels = kmeans.labels_;  # cluster labels
digits_centers = kmeans.cluster_centers_;  # centroid of clusters

pca = PCA(n_components=2);
pca.fit(digits.data);
digits_pc = pca.transform(digits.data);  #  data coordinates in the PC space
centers_pc = pca.transform(digits_centers);  # the cluster centroids in the PC space

fig = plt.figure(1);
plt.title('Clustering of the digits data after PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(digits_pc[:,0],digits_pc[:,1],s=50,c=digits_labels);  # data
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');  # centroids
plt.legend();
plt.show();
