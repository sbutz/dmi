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
    data['date'] = pd.to_datetime(data[['year','month','day']])

    idx = data.births.idxmax()
    print(data.loc[idx])

    data = data[data.year == data.loc[idx].year]

    plt.bar(data.date, data.births)
    plt.ylabel('Geburten')
    plt.show()

    """ Es gibt keine Besonderheiten. """


if __name__ == '__main__':
    main()