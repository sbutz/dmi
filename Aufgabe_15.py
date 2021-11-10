"""
Antwort:
Genauigkeiten zwischen ~50% und ~72%.
Ist das robust? Ja?
"""
# Vorlesung Data Mining
# Kapitel 3: Beispiel zu Iris-Bewertung mit oneR
# Edwin Schicker
import numpy as np
import pandas as pd
import sklearn as scn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Datensatz zu Iris laden
dataset = load_iris()
X = dataset["data"]                  # die gegebenen Daten
y = dataset["target"]                # die Zuordnung zu den 3 Iristypen
#print(dataset["DESCR"])              # ausführliche Beschreibung
n_daten, n_features = X.shape     # Zeilen (Daten) und Spalten (Features)

#Diskretisieren mit Hilfe des Mittelwerts
# Mittelwerte der vier Attribute bestimmen
mittel = X.mean(axis=0)
Xdiskr = np.array(X >= mittel, dtype='int')
features = np.array(["sepallength","sepalwidth","petallength","petalwidth"])

# in DataFrame alle Daten ablegen
df = pd.DataFrame(Xdiskr, columns=features)
df["iris"] = y                    # Iris-Typen in weiterer Spalte hinzufügen
print(df)

input("Return drücken")

# Wir splitten in Trainings- und Testsätze
zufallsstrom = 2             # Zufallsstrom setzen

df_train, df_test = train_test_split(df, random_state=zufallsstrom)
print("Es gibt {} Trainingssätze und {} Testsätze"
                               .format(df_train.shape[0],df_test.shape[0]))

input("Return drücken")

# für alle drei Iris-Typen (von 0 bis 2) die Anzahl der Treffer ermitteln, abhängig vom feature
def train(X, feature):
    werte = X[feature].unique()                     # alle auftretenden Werte merken
    vorhersagen = pd.Series(index=werte, dtype=int) # Feld für Vorhersagen  
    fehler = 0
    # in Schleife alle Vorhersagen und Fehler ermitteln
    for i in werte:
        iris, fehl = train_feature_wert(X, feature, i)
        vorhersagen[i] = iris
        fehler += fehl
    return vorhersagen, fehler

# für alle drei Iris-Typen (von 0 bis 2) die Anzahl der Treffer ermitteln, abhängig von feature und wert
def train_feature_wert(X, feature, wert):
    df_help = df_train[df_train[feature] == wert]        #nur Zeilen, deren Feature den Wert enthält
    iris_zaehler = df_help.iris.value_counts()           #zählt alle Iris Vorkommnisse
    imax = iris_zaehler.idxmax()         #häufigste Iris
    fehler = iris_zaehler.sum() - iris_zaehler.max()    #Anzahl der falschen Angaben: alle - max
    return imax, fehler

alle_vorhersagen = pd.DataFrame(columns=features)       # DataFrame mit allen Vorhersagen anlegen
fehler = pd.Series(dtype=int)                           # Fehlerfeld anlegen
for feature in features:                                    # Schleife über alle Features
    vorhersage, fehl = train(df_train, feature)             # Vorhersage berechnen
    alle_vorhersagen[feature] = vorhersage                  # Vorhersage abspeichern
    fehler[feature] = fehl                                  # ebenso die Fehler
# Wir ermitteln jetzt das Ergebnis mit den wenigsten Fehlern:
minfehler = fehler.min()
bestfeature = fehler.idxmin()
ergebnis = alle_vorhersagen[bestfeature]                # Zuordnung Wert --> Iristyp
print("Das beste Modell liefert die Untersuchung nach {}".format(bestfeature))
print("Dabei gilt folgende Vorhersage: Wert --> Iristyp:")
print(ergebnis)

input("Return drücken")

# Dieses Ergebnis wird anhand der Testdaten jetzt getestet
# Dazu wird eine Spalte prognose erzeugt, und mit Hilfe der Ergebnis-Daten werden
# die Feature-Werte umgewandelt in Iriswerte
df_test["prognose"] = df_test.apply(lambda x: ergebnis[x[bestfeature]], axis=1)

print ("Vergleich der Prognose mit den tatsächlichen Werten")
print(df_test[["prognose","iris"]])

input("Return drücken")

# Jetzt werden die Prognosewerte mit den tatsächlichen Werten verglichen
print("Die Genauigkeit ist {:.1f}%".format(100*np.mean(df_test.prognose == df_test.iris)))