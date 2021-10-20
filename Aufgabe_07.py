#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd

def main():
    df = pd.read_csv('US_births_2000-2014.csv',
        sep=',',
        dtype=int,
        header=0,
        names=['year','month','day','day_of_week','births'],
    )

    fig, axs = plt.subplots(nrows=1, ncols=2)

    axs[0].hist(df.births)

    data = []
    for y in range(2000,2003):#df.year.unique():
        data.append(df[df.year==y].births)
    axs[1].hist(data)

    plt.show()


if __name__ == '__main__':
    main()