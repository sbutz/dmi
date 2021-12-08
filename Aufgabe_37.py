#!/usr/bin/env python

"""
Antwort:
Ergebnisse varien teilweise stark.
Mit mehr Features viel besser.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('./auto-mpg.data',
        usecols=["mpg","cylinders","displacement","horsepower","acceleration","year","origin"],
        skipinitialspace=True,
        #comment="\t",
        sep='\s+',
        header=0,
    )
    df = df[df.horsepower != '?']
    df.horsepower = df.horsepower.astype(float)

    X = df.iloc[:,1:]
    X = X[["horsepower"]]
    y = df.iloc[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.3,
        random_state=4,
    )

    n = len(X.columns)
    model = Sequential()
    model.add(Dense(n, input_dim=n, activation='linear'))
    model.add(Dense(20, activation='selu'))
    model.add(Dense(1, activation='linear'))
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'],
    )

    model.fit(X_train, y_train,
        epochs=100,
        verbose=0,
        validation_data=(X_test, y_test),
    )

    plt.scatter(X_train, y_train, color='blue', marker='o', alpha=0.6, label='Training')
    plt.scatter(X_test, y_test, color='red', marker='o', alpha=0.6, label='Test')
    x = np.linspace(X.horsepower.min(), X.horsepower.max(), num=128)
    y = model.predict(x)
    plt.plot(x, y, color='green', label='nn')
    plt.show()


if __name__ == "__main__":
    main()