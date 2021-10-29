#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd

def main():
    facttab = pd.read_csv('bike_db/facttab.csv', sep=';')
    customer = pd.read_csv('bike_db/customer.csv', sep=';')

    facttab['totalprice'] = facttab.price * facttab.quantity

    df = pd.merge(left=facttab, right=customer, on='CSID')
    df = df.groupby(['CSID','custid','name']).sum().reset_index()
    df = df.sort_values('totalprice', ascending=False)
    print(df[['name', 'custid', 'totalprice']])


if __name__ == '__main__':
    main()