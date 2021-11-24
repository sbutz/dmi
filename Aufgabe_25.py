#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

def main():
    df = pd.read_csv('ionosphere.data', sep=',', header=None)
    X = df.iloc[:,:-1].to_numpy()
    y = df.iloc[:,-1:].to_numpy().flatten()

    for i in range(12, 18):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)

        model = GaussianNB()
        model.fit(X_train, y_train)
        probs = model.predict(X_test)
        konf = np.count_nonzero(y_test == probs) / y_test.size
        print(f'{i}: {konf}')
        plt.scatter(i, konf)
    plt.show()

    """
    Antwort:
    Ergebnis ähnlich
    """


if __name__ == '__main__':
    main()