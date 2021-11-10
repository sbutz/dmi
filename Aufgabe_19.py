"""
Antwort:
Ergebnisse f. 2 und 4 Klassen stimmen mit Erwartungen ueberein
Mit Random State, Elkan leicht mehr Iterationen noetig

"""
# Vorlesung Data Mining
# Kapitel 4: Beispiel zu k-Means mit Iris-Daten
# Edwin Schicker

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as scn; scn.set()
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Datensatz zu Iris laden und in DataFrame df speichern
dataset = load_iris()
X = dataset["data"]                  # die gegebenen Daten
y = dataset["target"]                # die Zuordnung zu den 3 Iristypen
features = np.array(["sepallength","sepalwidth","petallength","petalwidth"])
df = pd.DataFrame(X, columns=features)   # als DataFrame speichern

# Clusterbildung mit Hilfe von k-means
berechnung = KMeans(n_clusters=2, init='random', algorithm='elkan')
berechnung.fit(X)                       # Berechnung mit Hilfe der Daten aus DataFrame X
labels = berechnung.labels_             # errechnete Zugehörigkeit
centers = berechnung.cluster_centers_
print("Es wurden {} Iterationen durchgeführt".format(berechnung.n_iter_))
print(labels)
print(y)

# Zwei Bilder nebeneinander ausgeben: links ermittelte Cluster, rechts echte Zuordnung
bild = plt.figure(num="Irisdaten",figsize=(11,5))
inhalt1 = bild.add_subplot(121)         # Linkes Bild
inhalt1.scatter(df[labels==0].petallength, df[labels==0].petalwidth, c='b', s=25, label="Cluster1")      
inhalt1.scatter(df[labels==1].petallength, df[labels==1].petalwidth, c='y', s=25, label="Cluster2")      
inhalt1.scatter(df[labels==2].petallength, df[labels==2].petalwidth, c='g', s=25, label="Cluster3")      
inhalt1.scatter(centers[:,2], centers[:,3], c='black', s=60, alpha=0.7)
inhalt1.set_title("Mit k-Means berechnet")
inhalt1.axis([0,8,0,4])
inhalt1.legend()
inhalt1.set_xlabel("petallength")
inhalt1.set_ylabel("petalwidth")

inhalt2 = bild.add_subplot(122)         # Rechtes Bild
inhalt2.scatter(df[y==0].petallength, df[y==0].petalwidth, c='b', s=25, label="Setosa")      
inhalt2.scatter(df[y==1].petallength, df[y==1].petalwidth, c='y', s=25, label="Versicolor")      
inhalt2.scatter(df[y==2].petallength, df[y==2].petalwidth, c='g', s=25, label="Virginica")      
inhalt2.set_title("Originaldaten")
inhalt2.axis([0,8,0,4])
inhalt2.legend()
inhalt2.set_xlabel("petallength")
inhalt2.set_ylabel("petalwidth")

bild.tight_layout()       # verbessert die Ausgabe mit Abständen zwischen Subplots
plt.show()