
#!/usr/bin/env python3

import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd

def main():
    # Show all columns
    pd.set_option('display.max_columns', None)

    df = pd.read_csv('US_births_2000-2014.csv',
        sep=',',
        dtype=int,
        header=0,
        names=['year','month','day','day_of_week','births'],
    )

    pivot = pd.pivot_table(df,
        index='year',
        columns='month',
        values='births',
    )
    pivot.loc["Sum"] = pivot.sum()
    pivot["Max"] = pivot.max(axis=1)
    pivot["MaxMonat"] = pivot.idxmax(axis=1)

    print(pivot)
    #pivot.to_csv("pivot_aufgabe_05.csv")


if __name__ == '__main__':
    main()