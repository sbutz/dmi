#!/usr/bin/env python3

import pandas as pd

def main():
    facttab = pd.read_csv('bike_db/facttab.csv', sep=';')
    product = pd.read_csv('bike_db/product.csv', sep=';',
        usecols=['PSID', 'name'])
    customer= pd.read_csv('bike_db/customer.csv', sep=';',
        usecols=['CSID', 'name'])
    date    = pd.read_csv('bike_db/date.csv', sep=';')

    df = facttab.groupby(['CSID', 'PSID'], as_index=False).size()
    df = pd.merge(df, customer, on='CSID')
    df = pd.merge(df, product, on='PSID')
    df = df.sort_values(['CSID', 'size'], ascending=[True, False])
    df = df.groupby(['CSID']).head(3)
    print(df)

if __name__ == '__main__':
    main()