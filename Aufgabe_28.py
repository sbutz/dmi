#!/usr/bin/env python3

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

def main():
    df = pd.read_csv('diabetes.csv', sep=',', header=0)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    model = DecisionTreeClassifier(
        criterion='entropy',
        #splitter='best',
        #max_depth=15,
        min_samples_split=0.15,
        #min_samples_leaf=7,
        #max_features=7,
        min_impurity_decrease=0.02,
        #splitter='best')
    )
    model.fit(X_train, y_train)
    #plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
    #plt.show()

    y_predicted = model.predict(X_test)
    print(f'Korrektheit: {accuracy_score(y_test, y_predicted)}')
    #print(f'Trefferquote: {recall_score(y_test, y_predicted)}')
    #print(f'Praezision: {precision_score(y_test, y_predicted)}')


if __name__ == '__main__':
    main()