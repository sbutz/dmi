#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd

def main():
    data = pd.read_csv('US_births_2000-2014.csv',
        sep=',',
        dtype=int,
        header=0,
        names=['year','month','day','day_of_week','births'],
    )

    idx = data.births.idxmax()
    print(data.loc[idx])

    data = data[(data.year == data.loc[idx].year) &
        (data.month == data.loc[idx].month)]

    plt.bar(data.day, data.births)
    plt.xlabel(f'Tag in {data.loc[idx].month}/{data.loc[idx].year}')
    plt.ylabel('Geburten')
    plt.show()


if __name__ == '__main__':
    main()