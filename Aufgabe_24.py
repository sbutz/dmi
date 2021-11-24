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

    for i in range(12, 18):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)

        for j in range(1,21):
            model = KNeighborsClassifier(n_neighbors=j)
            model.fit(X_train, y_train)
            probs = model.predict(X_test)
            konf = np.count_nonzero(y_test == probs) / y_test.size
            print(f'{i}: {konf}')
            plt.scatter(j, konf)
    plt.show()

    """
    Antwort:
    Keine grossen Veraenderungen im Wertebereich der Konfidenz
    """


if __name__ == '__main__':
    main()