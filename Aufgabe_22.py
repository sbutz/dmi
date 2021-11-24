#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('ionosphere.data', sep=',', header=None)
    X = df.iloc[:,:-1].to_numpy()
    y = df.iloc[:,-1:].to_numpy().flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12)

    model = KNeighborsClassifier(n_neighbors=8)
    model.fit(X_train, y_train)
    probs = model.predict(X_test)
    konf = np.count_nonzero(y_test == probs) / y_test.size
    print(konf)

    """
    Das Ergebnis:
    Die Konfidenz ist der Anteil der richtigen Vorhersagen an den Vorhersagen.
    D.h. in {konf} Prozent der Fälle ist die Vorhersage korrekt.
    """


if __name__ == '__main__':
    main()