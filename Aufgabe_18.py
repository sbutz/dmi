#!/usr/bin/env python3

import seaborn
import matplotlib.pyplot as plt

"""
Antworten

a)
Sepal_width fuer alle sorten etwa gleich gross -> schlechtes unterscheidungsmerkmal

b)
petal length und petal_width sehr fast disjunkt fuer jede sorte

"""


def main():
    iris = seaborn.load_dataset("iris")
    seaborn.pairplot(iris, hue="species")
    plt.show()

if __name__ == "__main__":
    main()