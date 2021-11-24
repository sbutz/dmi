#!/usr/bin/env python3

"""
Antwort: Waehle Weather als erstes
"""

import numpy as np
import pandas as pd

def gini(p):
    return 1 - np.sum(p*p)

def entropy(p):
    return -np.sum(p*np.log2(p))

def main():
    df = pd.read_csv('weather.csv', sep=';', header=0)
    #print(df.Decision.unique()) # Cinema, Tennis, Stay in, Shopping
                                 # 0, 1, 2, 3

    #print(f'Features = {df.columns[1:-1]}') # Weather, Parents, Money

    p = df['Decision'].value_counts().values/df.shape[0]
    E = entropy(p)
    print(f'E = {E}')

    # Weather
    #print(df.Weather.unique()) # Sunny, Windy, Rainy
    Ef = 3/10 * entropy([1/3, 2/3]) + 4/10 * entropy([3/4, 1/4]) + 3/10 * entropy([2/3, 1/3])
    print(f'I(Weather) = {E-Ef}')

    # Parents
    #print(df.Parents.unique()) # Yes, No
    Ef = 5/10 * entropy([5/5]) + 5/10 * entropy([1/5, 2/5, 1/5, 1/5])
    print(f'I(Parents) = {E-Ef}')

    # Money
    #print(df.Money.unique()) # Rich, Poor
    Ef = 7/10 * entropy([3/7, 2/7, 1/7, 1/7]) + 3/10 * entropy([3/3])
    print(f'I(Money) = {E-Ef}')


if __name__ == '__main__':
    main()