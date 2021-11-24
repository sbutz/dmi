#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

def main():
    df = pd.read_csv('weather.csv', sep=';', header=0)
    map = {'Sunny': 0, 'Windy': 1, 'Rainy': 2}
    df['Weather'] = [map[v] for v in df['Weather']]
    map = {'Yes': 0, 'No': 1}
    df['Parents'] = [map[v] for v in df['Parents']]
    map = {'Rich': 0, 'Poor': 1}
    df['Money'] = [map[v] for v in df['Money']]
    X = df[['Weather', 'Parents', 'Money']]
    y = df['Decision']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = DecisionTreeClassifier(criterion='entropy', splitter='best')
    model.fit(X_train, y_train)
    plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
    plt.show()


if __name__ == '__main__':
    main()