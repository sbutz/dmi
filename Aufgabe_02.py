#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.core.frame import DataFrame

def main():
    data = pd.read_csv('US_births_2000-2014.csv',
        sep=',',
        dtype=int,
        header=0,
        names=['year','month','day','day_of_week','births'],
    )

    """ Teilaufgabe A """
    df = data.groupby('day_of_week')['births'].sum()
    domb = df.idxmax()
    print(f'Am {domb}. Wochentag werden die meisten Kinder geboren')
    plt.bar(df.index, df.values)
    plt.xlabel('Wochentag')
    plt.xlabel('Geburten')
    plt.show()

    """ Teilaufgabe B """
    df = data[data.day_of_week == domb].groupby('year')['births'].sum()
    plt.plot(df.index, df.values)
    plt.xlabel('Jahr')
    plt.ylabel('Geburten')
    plt.show()

    """ Teilaufgabe C """
    series = data.groupby(['year', 'month'])['births'].sum()
    df = pd.DataFrame(series).reset_index()
    g = sns.catplot(
        data=df,
        kind='bar',
        x='month',
        y='births',
        hue='year',
        ci='sd',
    )
    g.despine(left=True)
    g.set_axis_labels('', 'Geburten')
    plt.show()
    """
    Die meisten Geburten gibt es im August, die wenigsten im Februar.
    Gilt fuer fast alle Jahre.
    """


if __name__ == "__main__":
    main()