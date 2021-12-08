#!/usr/bin/env python

"""
Ergebnisse:
a) Trennzeichen \s+
b) usecols= hilft auch
c) 6x '?' in horsepower
d) astype(float)
e) 
f) car-name, model 
g) normalisierung noch nicht gelernt

"""

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import pandas.plotting

def main():
    df = pd.read_csv('./auto-mpg.data',
        usecols=["mpg","cylinders","displacement","horsepower","acceleration","model","year","origin"],
        skipinitialspace=True,
        #comment="\t",
        sep='\s+',
        header=0,
    )
    df = df[df.horsepower != '?']
    df.horsepower = df.horsepower.astype(float)

    # Scattermatrix
    pd.plotting.scatter_matrix(df,
        figsize=(15,15),
        marker='o',
        c=df.mpg.values,
        s = 30,
        alpha = 0.8,
    )
    plt.show()
    df = df.drop(columns=["model"])

    X = df.iloc[:,1:]
    y = df.iloc[:,0]

    print(X)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.3,
        random_state=4)

if __name__ == "__main__":
    main()