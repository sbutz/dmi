#!/usr/bin/env python

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('./automobile85.data', sep=',')
    df = df[df.price != '?']
    df.price = df.price.astype(float)
    X = df[['highway-mpg']]
    y = df[['price']]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.3,
        random_state=4)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    ax2.scatter(X_train, y_train, color='black', alpha=0.7, label="Trainingsdaten")
    ax2.scatter(X_test, y_test, color='blue', alpha=0.7, label="Testdaten")

    for d in range(2,11):
        poly = PolynomialFeatures(degree=d)
        X_train_p = poly.fit_transform(X_train)
        X_test_p = poly.fit_transform(X_test)

        model = LinearRegression()
        model.fit(X_train_p, y_train)

        ax1.bar(d, model.score(X_test_p, y_test))
        #print(f'{d} R Trainingsdaten: {model.score(X_train_p, y_train)}')
        #print(f'{d} R Trainingsdaten: {model.score(X_test_p, y_test)}')

        x = np.linspace(df['highway-mpg'].min(), df['highway-mpg'].max(), 100).reshape(-1, 1)
        x_p = poly.fit_transform(x)
        y = model.predict(x_p)
        ax2.plot(x, y, color=np.random.rand(3,), label=f'd={d}')

    ax2.legend()
    plt.show()


if __name__ == "__main__":
    main()