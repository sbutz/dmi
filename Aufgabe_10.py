#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd

def main():
    facttab = pd.read_csv('bike_db/facttab.csv', sep=';')
    product = pd.read_csv('bike_db/product.csv', sep=';')
    customer= pd.read_csv('bike_db/customer.csv', sep=';')
    date    = pd.read_csv('bike_db/date.csv', sep=';')

    product = product[product.prodgroup == 'Lady bicycle'][['PSID', 'artid', 'name']]
    date = date[date.month == 201902]['DSID']
    customer = customer[customer.state == 'Bayern']['CSID']

    df = pd.merge(right=facttab, left=date, on='DSID')
    df = pd.merge(right=df, left=customer, on='CSID')
    df = pd.merge(right=df, left=product, how='left', on='PSID')
    df = df.groupby(['PSID', 'artid', 'name']).sum().reset_index()
    print(df[['artid', 'name', 'quantity']])


if __name__ == '__main__':
    main()