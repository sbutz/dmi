#!/usr/bin/env python

"""
Ergebnis:
- Gewicht ist naehrungsweise lin. abh. von Width.
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas.plotting

def main():
    df = pd.read_csv('./Fish.csv', sep=',', header=0)

    # Scatter matrix
    #pd.plotting.scatter_matrix(df,
    #    figsize=(15,15),
    #    marker='o',
    #    c=df.Weight.values,
    #    s = 30,
    #    alpha = 0.8,
    #)
    #plt.show()

    # Linear Model fuer Width
    X = df[['Width']]
    y = df[['Weight']]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.3,
        random_state=4)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f'R2 Trainingsdaten: {model.score(X_train, y_train)}')
    print(f'R2 Testdaten: {model.score(X_test, y_test)}')

    # plot
    plt.scatter(X_train, y_train, color='black', alpha=0.7, label="Trainingsdaten")
    plt.scatter(X_test, y_test, color='blue', alpha=0.7, label="Testdaten")
    x = [[df['Width'].min()], [df['Width'].max()]]
    y = model.predict(x)
    plt.plot(x, y, color='red', label=f'linear')
    plt.show()

if __name__ == "__main__":
    main()