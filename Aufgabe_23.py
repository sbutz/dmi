#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

def main():
    df = pd.read_csv('ionosphere.data', sep=',', header=None)
    X = df.iloc[:,:-1].to_numpy()
    y = df.iloc[:,-1:].to_numpy().flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12)

    points = np.empty(shape=(20,1))
    for i in range(1, 21):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, y_train)
        probs = model.predict(X_test)
        konf = np.count_nonzero(y_test == probs) / y_test.size
        points[i-1] = konf
    plt.plot(np.arange(1,21), points)
    plt.show()

    """
    Schlussfolgerungen:
    Konfidenz zwischen 82 und 90 abh. von Test Daten
    Beste Konfidenz bei n_neighbors=1
    """


if __name__ == '__main__':
    main()