#!/usr/bin/env python

"""
Antwort:
Ergebnisse varien teilweise stark.
Mit mehr Features viel besser.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

def main():
    df = pd.read_csv('./auto-mpg.data',
        usecols=["mpg","cylinders","displacement","horsepower","acceleration","year","origin"],
        skipinitialspace=True,
        #comment="\t",
        sep='\s+',
        header=0,
    )
    df = df[df.horsepower != '?']
    df.horsepower = df.horsepower.astype(float)

    X = df.iloc[:,1:]
    #X = X[["horsepower"]]
    y = df.iloc[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.3,
        random_state=4,
    )

    model = Sequential()
    n = len(X.columns)
    model.add(Dense(n, input_dim=n, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.summary()

    hist = model.fit(X_train, y_train,
        epochs=100,
        batch_size=16,
        verbose=0,
        validation_data=(X_test, y_test),
    )

    result = model.evaluate(X_train, y_train)

    print("Trainingsdaten")
    print("{}: {:.2f}\n{}: {:.2f}".format(model.metrics_names[0], result[0],model.metrics_names[1], result[1]))
    result = model.evaluate(X_test, y_test)
    print("Testdaten")
    print("{}: {:.2f}\n{}: {:.2f}".format(model.metrics_names[0], result[0],model.metrics_names[1], result[1]))

    auswertung = pd.DataFrame.from_dict(hist.history)
    print("------ Lernen des Netzes plotten --------")
    fig = plt.figure(figsize=(20,8), num="Neuronales Netz")
    bild1 = fig.add_subplot(121)
    bild1.plot(auswertung.index, auswertung.iloc[:,2], color = 'blue', label="Training")    
    bild1.plot(auswertung.index, auswertung.iloc[:,0], color = 'red', label="Validierung")
    bild1.legend()    
    bild1.set_xlabel('epoch')
    bild1.set_ylabel(model.metrics_names[0])
    bild1.set_title("Neuronales Netz lernt: Loss-Kurve")
    bild2 = fig.add_subplot(122)
    bild2.plot(auswertung.index, auswertung.iloc[:,3], color = 'blue', label="Training")    
    bild2.plot(auswertung.index, auswertung.iloc[:,1], color = 'red', label="Validierung")    
    bild2.legend()    
    bild2.set_xlabel('epoch')
    bild2.set_ylabel(model.metrics_names[1])
    bild2.set_title("Neuronales Netz lernt: Accuracy-Kurve")
    plt.show()


if __name__ == "__main__":
    main()