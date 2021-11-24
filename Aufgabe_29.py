# Vorlesung Data Mining
# Kapitel 5: Test von SVM und linearer SVM
# Beispiel Iris 
# Edwin Schicker

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap

# Farbtafeln für 3 Klassen
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) #Hintergrund
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  #Plottpunkte

# Irisdaten laden
iris = load_iris()
X = iris.data
y = iris.target
features = np.array(["sepallength","sepalwidth","petallength","petalwidth"])
df = pd.DataFrame(X, columns=features)
X = df[["petallength", "petalwidth"]]

# "normales" SVM mit linearem Kernel und Lineares SVM
model_svm = svm.SVC(kernel="linear")
model_svm.fit(X, y);
model_svmlin = svm.LinearSVC()
model_svmlin.fit(X, y)
model_svmpoly2 = svm.SVC(kernel='poly', degree=2)
model_svmpoly2.fit(X, y)
model_svmpoly3 = svm.SVC(kernel='poly', degree=3)
model_svmpoly3.fit(X, y)

# Modell überprüfen mit 100x100 Bildpunkten (x,y)
x_values = np.linspace(0.9,7,100)   # erzeugt 100 x-Werte zwischen 0.9 und 7
y_values = np.linspace(0,3,100)     # erzeugt 100 y-Werte zwischen 0 und 3
# Erzeugen eines Gitters zur flächenweisen Ausgabe:
x_2, y_2 = np.meshgrid(x_values, y_values)   # x2 und y2 sind 2dim Felder (100x100)
# ravel wandelt 2 dim Feld in 1 Dimension um, 100x100 --> 10000
# np.c_ vereinigt beide 1dim Felder zu einem 2dim Feldpaar: wir haben 10000 Bildpunkte!
punkte = np.c_[x_2.ravel(), y_2.ravel()]  

proposal_svm = model_svm.predict(punkte)       # Vorhersage für alle 10000 Bildpunkte
proposal_svmlin = model_svmlin.predict(punkte) # Vorhersage für alle 10000 Bildpunkte
proposal_svmpoly2 = model_svmpoly2.predict(punkte) # Vorhersage für alle 10000 Bildpunkte
proposal_svmpoly3 = model_svmpoly3.predict(punkte) # Vorhersage für alle 10000 Bildpunkte

# Reshape ändert Form
proposal_svm = proposal_svm.reshape((100,100))  # Rückwandeln: 10000 --> 100x100
proposal_svmlin = proposal_svmlin.reshape((100,100))  # Rückwandeln: 10000 --> 100x100
proposal_svmpoly2 = proposal_svmpoly2.reshape((100,100))  # Rückwandeln: 10000 --> 100x100
proposal_svmpoly3 = proposal_svmpoly3.reshape((100,100))  # Rückwandeln: 10000 --> 100x100

# Und jetzt visualisieren
bild = plt.figure(num="Iris - Support Vector Machine", figsize=(12,10))

# Erst das Bild mit SVM
plot1 = bild.add_subplot(221)
# Flächenweise Ausgabe der Bildpunkte
plot1.pcolormesh(x_values, y_values, proposal_svm, cmap=cmap_light, shading="auto")
# Ausgabe der Irispunkte
plot1.scatter(X.petallength, X.petalwidth, c=y, cmap=cmap_bold)
plot1.set_xlabel("Petal Länge (cm)")
plot1.set_ylabel("Petal Weite (cm)")
plot1.set_title("Iris mit SVM (linearer Kern)")

# Erst das Bild mit SVM
plot2 = bild.add_subplot(222)
# Flächenweise Ausgabe der Bildpunkte
plot2.pcolormesh(x_values, y_values, proposal_svmlin, cmap=cmap_light, shading="auto")
# Ausgabe der Irispunkte
plot2.scatter(X.petallength, X.petalwidth, c=y, cmap=cmap_bold)
plot2.set_xlabel("Petal Länge (cm)")
plot2.set_ylabel("Petal Weite (cm)")
plot2.set_title("Iris mit Linearer SVM")

# Erst das Bild mit SVM
plot3 = bild.add_subplot(223)
# Flächenweise Ausgabe der Bildpunkte
plot3.pcolormesh(x_values, y_values, proposal_svmpoly2, cmap=cmap_light, shading="auto")
# Ausgabe der Irispunkte
plot3.scatter(X.petallength, X.petalwidth, c=y, cmap=cmap_bold)
plot3.set_xlabel("Petal Länge (cm)")
plot3.set_ylabel("Petal Weite (cm)")
plot3.set_title("Iris mit Polynomieller SVM (n=2)")

# Erst das Bild mit SVM
plot4 = bild.add_subplot(224)
# Flächenweise Ausgabe der Bildpunkte
plot4.pcolormesh(x_values, y_values, proposal_svmpoly3, cmap=cmap_light, shading="auto")
# Ausgabe der Irispunkte
plot4.scatter(X.petallength, X.petalwidth, c=y, cmap=cmap_bold)
plot4.set_xlabel("Petal Länge (cm)")
plot4.set_ylabel("Petal Weite (cm)")
plot4.set_title("Iris mit Polynomieller SVM (n=3)")

plt.show()