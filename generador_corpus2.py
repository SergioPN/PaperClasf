# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

#%%

from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from scipy.stats import entropy

import numpy as np

mypath = "D:\ml2\BD\miguel2\papers\\"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

papers = []
tfidf = []
clases = ["C","A","C","C","L","C","C","L","L","L","L","L","C","A","L","C","L","C","L","A"] #"Arte, Letras, Ciencia"
clases = ["C","L","C","C","L","C","C","L","L","L","L","L","C","L","L","C","L","C","L","L"] #
n_clases = list(map(lambda x: 1 if x == 'C' else 0, clases))
         
         
for file in onlyfiles:
    f = open(mypath + str(file))
    lines = f.read()
    str_text = str(lines)
    str_lines = re.sub(" +", " ", str_text).strip().replace(",","").replace("\\","").replace("-","").replace("'s","")
    str_lines = re.sub("[0-9]+", "", str_lines).replace("\n", "").replace(". ",".")
    str_lines = re.sub(" +", " ", str_lines)
    str_lines = str_lines[:str_lines.find("PROGRAM")].strip().lower()
    papers.append(str_lines)
    

#Tfid = TfidfVectorizer()
#vec = Tfid.fit_transform(papers)
vec = CountVectorizer().fit_transform(papers)
vec = TfidfTransformer().fit_transform(vec)
mat = vec.toarray()
mat = mat[:]
feature_names = Tfid.get_feature_names()
df_mat = pd.DataFrame(mat, columns=feature_names, index=onlyfiles)

entropias = entropy(mat)
pd.Series(entropias)
orden = np.argsort(entropias)
mat[0,orden]
words_entropy = np.array(feature_names)[orden]
df_mat_entropy = pd.DataFrame(mat[:,orden], columns = words_entropy, index=onlyfiles)

#%%
X_train, X_test, y_train, y_test = train_test_split(mat, clases, test_size=0.15, random_state = 1)


svm_clf = SVC(C = 1, gamma='auto', kernel='linear')
svm_clf.fit(X_train, y_train)
y = svm_clf.predict(X_test)
print(classification_report(y_test, y))

entropias = entropy(mat)
pd.Series(entropias)
orden = np.argsort(entropias)
mat[0,orden]
words_entropy = np.array(feature_names)[orden]
df_mat_entropy = pd.DataFrame(mat[:,orden], columns = words_entropy, index=onlyfiles)


#X_train, X_test, y_train, y_test = train_test_split(mat[:][:,orden[:20]], clases, test_size=0.15, random_state = 1)
X_train, X_test, y_train, y_test = train_test_split(df_mat_entropy.iloc[:,:200], clases, test_size=0.15, random_state = 1)


svm_clf = SVC(C = 1, gamma='auto', kernel='linear')
svm_clf.fit(X_train, y_train)
y = svm_clf.predict(X_test)
print(classification_report(y_test, y))

#gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)


#%% Clustering


from sklearn import cluster
from sklearn import manifold
from sklearn import metrics
import seaborn as sns

Train = df_mat_entropy.iloc[:,:200]
Train = manifold.TSNE().fit_transform(Train)

km = cluster.KMeans(n_clusters = 2, max_iter = 1000, algorithm="full")
y = km.fit_predict(Train)
plt.figure()
plt.subplot(121)
plt.title("200 Muestras")
plt.scatter(np.arange(len(y)), y, color="red", label="KMeans")
plt.scatter(np.arange(len(n_clases)), n_clases, edgecolors="green", facecolors='none', linewidths=2, label = "Real")
plt.legend(loc='best')
plt.subplot(122)
sns.heatmap(metrics.confusion_matrix(n_clases,y)/len(n_clases))
print(metrics.homogeneity_score(y,clases))
print(metrics.classification_report(n_clases, y))
plt.show()



Train = df_mat_entropy.iloc[:,:100]
Train = manifold.TSNE().fit_transform(Train)

km = cluster.KMeans(n_clusters = 2, max_iter = 1000, algorithm="full")
y = km.fit_predict(Train)
plt.figure()
plt.subplot(121)
plt.title("100 Muestras")
plt.scatter(np.arange(len(y)), y, color="red", label="KMeans")
plt.scatter(np.arange(len(n_clases)), n_clases, edgecolors="green", facecolors='none', linewidths=2, label = "Real")
plt.legend(loc='best')
plt.subplot(122)
sns.heatmap(metrics.confusion_matrix(n_clases,y)/len(n_clases))
print(metrics.homogeneity_score(y,clases))
print(metrics.classification_report(n_clases, y))
plt.show()


Train = df_mat_entropy.iloc[:,:50]
Train = manifold.TSNE().fit_transform(Train)


km = cluster.KMeans(n_clusters = 2, max_iter = 1000, algorithm="full")
y = km.fit_predict(Train)
plt.figure()
plt.subplot(121)
plt.title("50 Muestras")
plt.scatter(np.arange(len(y)), y, color="red", label="KMeans")
plt.scatter(np.arange(len(n_clases)), n_clases, edgecolors="green", facecolors='none', linewidths=2, label = "Real")
plt.legend(loc='best')
plt.subplot(122)
sns.heatmap(metrics.confusion_matrix(n_clases,y)/len(n_clases))
print(metrics.homogeneity_score(y,clases))
print(metrics.classification_report(n_clases, y))
plt.show()


scores = []
for n in np.arange(10,630,30):
    Train = df_mat_entropy.iloc[:,:n]
    Train = manifold.TSNE().fit_transform(Train)
    
    km = cluster.KMeans(n_clusters = 2, max_iter = 1000, algorithm="full")
    y = km.fit_predict(Train)
#    scores = np.append(scores, metrics.completeness_score(y,clases))
    scores = np.append(scores, km.score(Train))
    
plt.plot(np.arange(10,630,30),scores, 'o-')


