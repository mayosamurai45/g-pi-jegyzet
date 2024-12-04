#%%
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split #, cross_validate
from sklearn.decomposition import PCA
import matplotlib.colors as col;  # importing coloring tools from MatPlotLib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import ConfusionMatrixDisplay; #  importing performance metrics
from sklearn.cluster import KMeans; # Class for K-means clustering
from sklearn.metrics import davies_bouldin_score;  # function for Davies-Bouldin goodness-of-fit 

import numpy as np
#%%
# 1. Olvassa be a breast_cancer beépített adatállományt és írassa ki a legfontosabb jellemzőit, úm. rekordok száma, attribútumok száma és osztályok száma
myCancer = load_breast_cancer()

rekordok_szama=myCancer.data.shape[0]
attributumok_szama=myCancer.data.shape[1]
osztalyok_szama=myCancer.target_names.shape[0]

print(f"Rekordok szama: {rekordok_szama}\nAttributumok szama: {attributumok_szama}\nOsztalyok szama: {osztalyok_szama}")

#%%
# 2. Készítsen pontdiagramot, ahol a tengelyeket a változók közül lehet megadni. Az aktuális változó nevek kerüljenek kiírásra a tengelyekre
# Default axis
x_axis = 0;  # x axis attribute (0,1,2,3)
y_axis = 1;  # y axis attribute (0,1,2,3)
# Enter axis from consol
user_input = input('X axis [0..30, default:0]: ');
if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=3 :
    x_axis = np.int8(user_input);
user_input = input('Y axis [0..30, default:1]: ');
if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=3 :
    y_axis = np.int8(user_input);    
colors = ['blue','red']; # colors for target values
fig = plt.figure(1);
plt.title('Scatterplot for iris dataset');
plt.xlabel(myCancer.feature_names[x_axis]);
plt.ylabel(myCancer.feature_names[y_axis]);
plt.scatter(myCancer.data[:,x_axis],myCancer.data[:,y_axis],s=50,c=myCancer.target,cmap=col.ListedColormap(colors));
plt.show();


#%%
# 3. Particionálja az adatállományt 70% tanító és 30% tesztállományra. Keverje össze a rekordokat és a véletlenszám-generátort inicializálja az idei évvel.
X_train, X_test, y_train, y_test = train_test_split(myCancer.data, myCancer.target, shuffle=True, test_size=0.3, random_state=2023);

#%%
# 4. Végezzen felügyelt tanítást az alábbi modellekkel és beállításokkal: 
    #döntési fa (5 mélység, Gini homogenitási kritérium), 
    #logisztikus regresszió (ha szükséges akkor növelje meg az iteráció számot) és 
    #neurális háló (1 rejtett réteg 5 neuronnal, logisztikus aktivációs függvény). 
    #A teszt score alapján hasonlítsa össze az illesztett modelleket, melyeket nyomtasson ki.

#Döntési fa
class_tree = DecisionTreeClassifier(criterion='gini',max_depth=5);

class_tree.fit(X_train, y_train);
score_train_tree = class_tree.score(X_train, y_train); # Goodness of tree on training dataset
score_test_tree = class_tree.score(X_test, y_test); # Goodness of tree on test dataset

#Logisztikus regresszió
reg = LogisticRegression(max_iter=10000)

reg.fit(X_train, y_train);
score_train_logreg = reg.score(X_train, y_train)
score_test_logreg = reg.score(X_test, y_test);

#Neurális Háló
neural_classifier = MLPClassifier(hidden_layer_sizes=(5),activation='logistic',max_iter=500);  #  number of hidden neurons: 5

neural_classifier.fit(X_train,y_train);
score_train_neural = neural_classifier.score(X_train,y_train);  #  goodness of fit
score_test_neural = neural_classifier.score(X_test,y_test);  #  goodness of fit
ypred_neural = neural_classifier.predict(X_test);   # spam prediction

print(f"Dontesi fa pontszama: {score_test_tree}")
print(f"Logisztikus regresszio pontszama: {score_test_logreg}")
print(f"Neuralis halo pontszama: {score_test_neural}")

#%%
# 5. Ábrázolja a 4. pontban eredményül kapott döntési fát.

fig = plt.figure(1,figsize = (16,10),dpi=100);
plot_tree(class_tree, feature_names = myCancer.feature_names.tolist(), 
               class_names = myCancer.target_names.tolist(),
               filled = True, fontsize = 6);
#%%
# 6. Számolja ki a 4. pont legjobb modelljére a teszt tévesztési mátrixot, amelyet jelenítsen is meg egyben.
# Plotting non-normalized confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, reg.predict(X_test), display_labels=myCancer.target_names.tolist())
plt.show()

#%%
# 7. Végezzen nemfelügyelt tanítást a K-közép módszerrel a tanító állomány input attribútumain. Határozza meg az optimális klaszterszámot 30-ig való kereséssel a DB index alapján a teszt állományon.
Max_K = 30;  # maximum cluster number
SSE = np.zeros((Max_K-2));  #  array for sum of squares errors
DB = np.zeros((Max_K-2));  # array for Davies Bouldin indeces
n_c=2;
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2020);
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
#%%
# 8. K=3 klaszterszám mellett vizualizálja a klasztereket egy pontdiagramon, ahol a két koordináta egy 2 dimenziós PCA eredménye.
kmeans = KMeans(n_clusters=3, random_state=2023);  # instance of KMeans class
kmeans.fit(myCancer.data);   #  fitting the model to data
myCancer_labels = kmeans.labels_;  # cluster labels
myCancer_centers = kmeans.cluster_centers_;  # centroid of clusters

pca = PCA(n_components=2);
pca.fit(myCancer.data);
myCancer_pc = pca.transform(myCancer.data);  #  data coordinates in the PC space
centers_pc = pca.transform(myCancer_centers);  # the cluster centroids in the PC space

fig = plt.figure(1);
plt.title('Clustering of the breast cancer data after PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(myCancer_pc[:,0],myCancer_pc[:,1],s=50,c=myCancer_labels);  # data
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');  # centroids
plt.legend();
plt.show();