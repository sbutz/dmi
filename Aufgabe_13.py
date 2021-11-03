#!/usr/bin/env python3

import pandas as pd

def main():
    facttab = pd.read_csv('bike_db/facttab.csv', sep=';',
        usecols=['PSID', 'ordid'])
    product = pd.read_csv('bike_db/product.csv', sep=';',
        usecols=['PSID', 'name'])
    customer= pd.read_csv('bike_db/customer.csv', sep=';',
        usecols=['CSID', 'name'])
    date    = pd.read_csv('bike_db/date.csv', sep=';')

    df = pd.merge(facttab, facttab, on='ordid')
    df = df.groupby(['PSID_x', 'PSID_y'], as_index=False).size()
    df['konf'] = df.apply(
        lambda row: row['size'] /
            df[(df.PSID_x==row['PSID_x']) & (df.PSID_y==row['PSID_x'])].iloc[0]['size'],
        axis=1,
    )
    df = df[df.konf < 1]
    df = df.sort_values('konf', ascending=False)
    df = df.head(5)

    print(df)

if __name__ == '__main__':
    main()