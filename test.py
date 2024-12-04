# 1. Olvassa be a digits beépített adatállományt és írassa ki a legfontosabb 
#   jellemzőit (rekordok száma, attribútumok száma és osztályok száma). (3 pont)
# 2. Készítsen többdimenziós vizualizációt a mátrix ábra segítségével (pairplot). (4 pont)
# 3. Particionálja az adatállományt 80% tanító és 20% tesztállományra.
#   Keverje össze a rekordokat és a véletlenszám-generátort inicializálja az idei évvel. (3 pont)
# 4. Végezzen felügyelt tanítást az alábbi modellekkel és beállításokkal:
#   döntési fa (4 mélység, entrópia homogenitási kritérium), logisztikus regresszió
#   (liblinear solverrel) és neurális háló (1 rejtett réteg 4 neuronnal, 
#   logisztikus aktivációs függvény). A teszt score alapján hasonlítsa össze az 
#   illesztett modelleket, melyeket nyomtasson ki. (10 pont)
# 5. Számolja ki az 5. pont legjobb modelljére a teszt tévesztési mátrixot. (4 pont)
# 6. Ábrázolja a tévesztési mátrixot. (3 pont)
# 7. Végezzen nemfelügyelt tanítást a K-közép módszerrel az input attribútumokon.
#   Határozza meg az optimális klaszterszámot 30-ig a DB indexszel. Az optimális
#   klaszterszám mellett vizualizálja a klasztereket egy pontdiagrammon, ahol a 
#   két koordináta egy 2 dimenziós PCA eredménye. (13 pont)


#%%
import os

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as c
import seaborn as sn

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split #, cross_validate
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay, davies_bouldin_score
#from scipy.special import expit

from sklearn.datasets import load_digits

import numpy as np
from numpy.random import default_rng
from datetime import datetime

RANDOM_SEED=datetime.today().year
r = default_rng(seed=RANDOM_SEED)

pd.options.display.max_columns = None

#%%
# Load data
_d = load_digits(as_frame=True)
d = _d.frame
d.target = _d.target.astype('category')

x, x_test, y, y_test = train_test_split(d.iloc[:,0:-1], d.iloc[:,-1], test_size=0.2, shuffle=True, random_state=RANDOM_SEED)

#%%
# Evaluate data
d.info()
print("=======")
print(f"Rekordok, attributumok (target nélkül): {d.shape[0],d.shape[1]-1}")
print(f"Osztalyok, (szám): {d.target.cat.categories}, {(len(d.target.cat.categories))}")
#%%
def sample_images(d):
    return np.hstack(list(x.array.reshape(8,-1) for k,x in
               d.iloc[r.choice(d.shape[0], size=10),:-1].iterrows()))
plt.imshow(sample_images(d))

# Mégis mit hogy pairplottolni egy 64 paraméteres adathalmazt? Ki se bírja számolni a gépem.
#sn.pairplot(d.iloc[1:3,:-1])

#%%
# Fit data

log = LogisticRegression(random_state=RANDOM_SEED, 
    solver='liblinear',).fit(x, y)
nn = MLPClassifier(random_state=RANDOM_SEED,
    hidden_layer_sizes=(4,),
    activation='logistic').fit(x, y)
tree = DecisionTreeClassifier(random_state=RANDOM_SEED,
    criterion='entropy',
    max_depth=4).fit(x, y)

# I assume this score is nonnegative
km, km_score = None, 0
_y_test = y_test.array[:,np.newaxis]
print("iterating kmeans", end="")
for clusters in range(2,31):
    print(".", end="")
    _km = KMeans(n_init='auto', n_clusters=clusters, random_state=RANDOM_SEED).fit(x_test)
    if (score := davies_bouldin_score(_y_test, _km.labels_)) > km_score:
        km, km_score = _km, score
print()

#%%
# Evaluate fit
models = {"log": log, "nn": nn, "tree": tree}
scores = list(((k, model), model.score(x_test, y_test)) for k, model in models.items())

print(f"model scores:\n" + "\n".join(f"{k[0]}: {s:.04}" for k,s in scores))

best_score, best_model = sorted(((x, k[1]) for k,x in scores), reverse=True)[0]

ConfusionMatrixDisplay.from_estimator(best_model, x_test, y_test)

# Ez erre az adathalmazra nem hiszem hogy túlságosan értelmes a PCA?...
pca = PCA(2, random_state=RANDOM_SEED).fit(d.iloc[:,:-1]);
d_pc = pd.DataFrame(pca.transform(x_test));
d_pc["klass"] = km.predict(x_test)
d_pc.klass = d_pc.klass.astype("category")
sn.relplot(d_pc, x=0, y=1, hue="klass")
_groups = d_pc.groupby(by="klass").groups
for c in d_pc.klass.cat.categories:
    plt.imshow(sample_images(d.iloc[_groups[c],:]))