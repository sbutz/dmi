#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd

def main():
    facttab = pd.read_csv('bike_db/facttab.csv', sep=';')
    product = pd.read_csv('bike_db/product.csv', sep=';')

    psid = product[product.artid == 100013].iloc[0]['PSID']
    orders = facttab[facttab.PSID == psid]['ordid'].unique()
    orders = facttab[(facttab.ordid.isin(orders) & (facttab.PSID != psid))]

    df = pd.merge(left=orders, right=product, on='PSID')
    df = df.groupby(['PSID', 'artid', 'name']).sum().reset_index()
    df = df.sort_values('quantity', ascending=False)
    print(df[['name', 'artid', 'quantity']].head(10))


if __name__ == '__main__':
    main()