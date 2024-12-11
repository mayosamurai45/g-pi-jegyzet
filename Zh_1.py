import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.metrics import davies_bouldin_score

# 1. feladat
# Load the data
url = "https://arato.inf.unideb.hu/ispany.marton/MachineLearning/Datasets/banknote_authentication.txt"
attribute_names = ["Variance", "skewness", "curtosis", "entropy", "target"]
data = pd.read_csv(url, header=None, names=attribute_names)

# Display data characteristics
print(f"Rekordok száma: {data.shape[0]}")
print(f"Attribútumok száma: {data.shape[1] - 1}")
print(f"Osztályok száma: {data['target'].nunique()}")

# 2. feladat
# Create a DataFrame and plot parallel coordinates
plt.figure(figsize=(10, 6))
parallel_coordinates(data, 'target', color=['blue', 'red'])
plt.title("Parallel Coordinates Plot")
plt.xlabel("Attributes")
plt.ylabel("Values")
plt.grid()
plt.show()

# 3. feladat
# Split the data
X = data.drop('target', axis=1)
y = data['target']
random_seed = 2024
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_seed, shuffle=True
)

# 4. feladat
# Train models
# Decision Tree
dt_model = DecisionTreeClassifier(max_depth=6, criterion='gini', random_state=random_seed)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

# Logistic Regression
lr_model = LogisticRegression(random_state=random_seed)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

# Gaussian Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)

# Compare accuracies
best_model = None
if dt_accuracy >= max(lr_accuracy, nb_accuracy):
    best_model = dt_model
    best_model_name = "Decision Tree"
    best_accuracy = dt_accuracy
elif lr_accuracy >= max(dt_accuracy, nb_accuracy):
    best_model = lr_model
    best_model_name = "Logistic Regression"
    best_accuracy = lr_accuracy
else:
    best_model = nb_model
    best_model_name = "Naive Bayes"
    best_accuracy = nb_accuracy

print(f"Legjobb modell: {best_model_name} teszt pontosság: {best_accuracy:.4f}")

# 5. feladat
# Compute and display confusion matrix for the best model
best_predictions = best_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, best_predictions)
print("Tévesztési mátrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 6. feladat
# ROC Curve for the best model
if best_model_name == "Logistic Regression" or best_model_name == "Naive Bayes":
    y_prob = best_model.predict_proba(X_test)[:, 1]
else:
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

if y_prob is not None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# 7. feladat
# Unsupervised learning with K-Means
db_scores = []
for k in range(2, 31):
    kmeans = KMeans(n_clusters=k, random_state=random_seed)
    kmeans.fit(X_train)
    db_index = davies_bouldin_score(X_test, kmeans.predict(X_test))
    db_scores.append((k, db_index))

optimal_k = min(db_scores, key=lambda x: x[1])[0]
print(f"Optimális klaszterszám: {optimal_k}")

# 8. feladat
# Visualize clusters with K=2 using PCA
kmeans = KMeans(n_clusters=2, random_state=random_seed)
kmeans.fit(X_train)
labels = kmeans.predict(X_test)

pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=labels, cmap="coolwarm", alpha=0.6)
plt.title("Clusters Visualized with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.show()
