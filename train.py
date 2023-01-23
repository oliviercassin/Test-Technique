#%%
# Avant de commencer a entrainer le model il faut d'abord explorer notre dataset
# afin de prendre connaissance de nos données.

import pandas as pd
import numpy as np

df = pd.read_csv('dataset.csv')
df.head(10)
df.info()
df.describe()

# On sait maintenant que notre df a une forme(900,5) avec 4 colonnes numerique et une 
# colonne de string. On sait aussi qu'il n'y a pas de valaurs manquantes dans le df.

#%%
# Ici on va plot les données afin de voir si visuellement on peut déjà separer les 
# les produits en 3 groupes distincts. On va aussi afficher la matrice de corrélations
# pour voir la dépendance entre les variables.

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='activity')

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)

# Il est difficile de faire 3 groupes distincts a l'oeil nue, et la matrice de corrélation
# nous a montré que les deux catégories les plus corrélées sont height et depth.

#%%
# C'est la première étape de la classification. On commence par séparer le df en deux,
# d'un coté les parametre que l'on va utiliser (height, weight, depth, width) et de 
# l'autre la classe que l'on veut predire (acticity). 

X = df[['height','width','depth','weight']]
y = df[['activity']]

#%%
# On importe ici les fonctions et les modèles que l'on va utiliser.

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

#%%
# On sépare les données en deux les données d'entrainement avec que lesquelles on va 
# entrainer le modèles et les données de test pour tester les performances des modèles.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#%%
# Ici j'ai choisi de tester 3 modèles de classification différents.

models=[]

dtc = tree.DecisionTreeClassifier()
models.append(dtc)

rfc = RandomForestClassifier()
models.append(rfc)

knn = KNeighborsClassifier()
models.append(knn)

#%%
# Maintenant on entraine chaque modèle sur les données d'entrainement avant des les tester
# sur les données de test. Ensuite on affiche des mesures de performance pour voir quel 
# modèle a été le plus performant.

for i in models:
    i.fit(X_train, y_train.values.ravel())
    y_pred = i.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy of %s is %s"%(i, acc))
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix of %s is \n%s"%(i, cm))
    print("\n",classification_report(y_test, y_pred),"\n")

# On peut voir grace à l'accuracy, la matrice de confusion et au rapport de classification
# que le RandomForestClassifier est le meilleur modèle.