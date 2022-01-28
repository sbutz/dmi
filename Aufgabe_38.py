#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def main():
    """ Datensatz laden """
    df = load_iris(as_frame=True)
    X = df.data.rename(columns={
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    })
    X = X[["petal_length", "petal_width"]]
    y = df.target
    y_cat = tf.keras.utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat,
        test_size=0.3,
        random_state=4,
    )



    """ Model erstellen und trainieren """
    model = Sequential()
    model.add(Dense(12, input_dim=len(X.columns), activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics='accuracy',
    )

    hist = model.fit(X_train, y_train,
        epochs=200,
        #batch_size=32,
        verbose=0,
        validation_data=(X_test, y_test),
    )



    """ Model bewerten"""
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



    """ Vorhersagekurve plotten """
    X0 = X[y==0]
    X1 = X[y==1]
    X2 = X[y==2]
    plt.scatter(X0["petal_length"], X0["petal_width"], color='red', marker='o', alpha=0.9)
    plt.scatter(X1["petal_length"], X1["petal_width"], color='green', marker='o', alpha=0.9)
    plt.scatter(X2["petal_length"], X2["petal_width"], color='blue', marker='o', alpha=0.9)

    x, y = np.mgrid[0:7:0.05, 0:3:0.05]
    X = np.vstack((x.flatten(), y.flatten())).T
    y = model.predict(X)
    y = np.argmax(y, axis=1)
    X0 = X[y==0]
    X1 = X[y==1]
    X2 = X[y==2]
    plt.scatter(X0[:,0], X0[:,1], color='red', marker='s', alpha=0.1)
    plt.scatter(X1[:,0], X1[:,1], color='green', marker='s', alpha=0.1)
    plt.scatter(X2[:,0], X2[:,1], color='blue', marker='s', alpha=0.1)

    plt.show()


if __name__ == "__main__":
    main()