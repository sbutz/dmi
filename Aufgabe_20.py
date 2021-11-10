#!/usr/bin/env python3

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    dataset = load_iris()
    X = dataset["data"]
    y = dataset["target"]
    features = np.array(["sepallength","sepalwidth","petallength","petalwidth"])
    df = pd.DataFrame(X, columns=features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12)
    model = GaussianNB()
    model.fit(X_train, y_train)
    labels = model.predict(X)

    y_predicted = model.predict(X_test)
    precision = 1 - np.count_nonzero(y_test-y_predicted) / X_test.size
    print(f'Genauigkeit: {precision}')


    # Zwei Bilder nebeneinander ausgeben: links ermittelte Cluster, rechts echte Zuordnung
    bild = plt.figure(num="Irisdaten",figsize=(11,5))
    inhalt1 = bild.add_subplot(121)         # Linkes Bild
    inhalt1.scatter(df[labels==0].petallength, df[labels==0].petalwidth, c='b', s=25, label="Cluster1")      
    inhalt1.scatter(df[labels==1].petallength, df[labels==1].petalwidth, c='y', s=25, label="Cluster2")      
    inhalt1.scatter(df[labels==2].petallength, df[labels==2].petalwidth, c='g', s=25, label="Cluster3")      
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


if __name__ == "__main__":
    main()