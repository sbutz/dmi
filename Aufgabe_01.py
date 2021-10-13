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
    data['season'] = data['date'].transform(day_to_season)


    """ Einbruch am Wochenende """
    print(data.groupby('day_of_week')['births'].sum())

    """ Niedrige Geburtenrate im Winter """
    print(data.groupby('season')['births'].sum())

    """ Hoehere Geburtenrate in 2006-2008 """
    print(data.groupby('year')['births'].sum())


    """ Plotten """
    data = data[data.year==2010]

    plt.plot(data['date'], data['births'])
    plt.xlabel("Datum")
    plt.ylabel("Geburten")
    plt.show()


def day_to_season(date):
    doy = date.timetuple().tm_yday
    if doy > 355:
        return 'winter';
    elif doy > 264:
        return 'fall'
    elif doy > 172:
        return 'summer'
    elif doy > 80:
        return 'spring'
    else:
        return 'winter'


if __name__ == "__main__":
    main()