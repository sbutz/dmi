#!/usr/bin/env python3

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
    print(pivot)

    pivot = pd.pivot_table(df,
        index='year',
        columns='day_of_week',
        values='births',
    )
    print(pivot)


if __name__ == '__main__':
    main()