#!/usr/bin/env python

"""
Ergebnis:
- Linear: Wesentlich besser
- Polynomiell: verdaechtig gut (overfitting?)
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('./Fish.csv', sep=',', header=0)

    X = df[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
    y = df[['Weight']]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.3,
        random_state=4)

    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f'Linear: R2 Trainingsdaten: {model.score(X_train, y_train)}')
    print(f'Linear: R2 Testdaten: {model.score(X_test, y_test)}')

    poly = PolynomialFeatures(degree=2)
    X_train_p = poly.fit_transform(X_train)
    X_test_p = poly.fit_transform(X_test)
    model_poly = LinearRegression()
    model_poly.fit(X_train_p, y_train)
    print(f'Polynomial: R2 Trainingsdaten: {model_poly.score(X_train_p, y_train)}')
    print(f'Polynomial: R2 Testdaten: {model_poly.score(X_test_p, y_test)}')

if __name__ == "__main__":
    main()