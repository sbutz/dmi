#!/usr/bin/env python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

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
    y = df.iloc[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.3,
        random_state=4,
    )

    model = Sequential()
    n = len(X.columns)
    model.add(Dense(n, input_dim=n, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

if __name__ == "__main__":
    main()