#!/usr/bin/env python3

"""
Datensatz:
- Trennzeichen ,
- Kein Header
- 351 Datensätze
- 35 Features, 1 boolean, 33 continuous numeric, 1 boolean
- Ausreißer nur -1, Z. 136

"""

import pandas as pd

def main():
    df = pd.read_csv('ionosphere.data', sep=',')

    print(df)

if __name__ == '__main__':
    main()