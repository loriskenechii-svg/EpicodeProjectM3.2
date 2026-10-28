#1. DATA EXPLORATION & PREPROCESSING

from sklearn.datasets import load_linnerud
from sklearn.preprocessing import StandardScaler
import pandas as pd

#Caricamento e Analisi del dataset
linnerud = load_linnerud(as_frame=True)

X = linnerud.data
Y = linnerud.target

df = pd.concat([X, Y], axis=1)

print(df.head())
print(df.info())
print(df.describe())

# Standardizzazione delle feature
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)


#2. TARGET ENGINEERING
scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y)

#PCA
from sklearn.decomposition import PCA

pca_Y = PCA(n_components=1)
Y_pca = pca_Y.fit_transform(Y_scaled)

#Target Waist
Y_scaled_df = pd.DataFrame(Y_scaled, columns=Y.columns)
y_waist = Y_scaled_df['Waist']

#3. MODEL TRAINING & EVALUATION

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

#Definizione e inizializzazione dei modelli
modelli = {
    'Linear Regression': LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

#Scenari
scenari = {
    "PCA": Y_pca,
    "Waist": y_waist
}

#Addestramento e valutazione dei modelli
risultati = []

for scenario, y in scenari.items():
    for nome, modello in modelli.items():

        modello.fit(X_scaled, y)
        predizioni = modello.predict(X_scaled)

        mse = mean_squared_error(y, predizioni)
        r2 = r2_score(y, predizioni)

        risultati.append([scenario, nome, mse, r2])

#4. ANALISI E VISUALIZZAZIONE DEI RISULTATI

import matplotlib.pyplot as plt

#Applicazione la PCA al set di feature
pca_X = PCA(n_components=2)
X_pca = pca_X.fit_transform(X_scaled)

#Creazione grafico
X_pc1 = X_pca[:, [0]]

plt.figure(figsize=(9, 6))

plt.scatter(X_pc1[:,0], Y_pca[:,0], label="Dati")

for nome, modello in modelli.items():
    modello.fit(X_pc1, Y_pca)
    predizioni = modello.predict(X_pc1)

    r2 = r2_score(Y_pca, predizioni)

    plt.plot(X_pc1[:, 0], predizioni, label=f"{nome} (R²={r2:.3f})")

plt.xlabel("PC1")
plt.ylabel("Target PCA")
plt.title("Confronto modelli di regressione")
plt.legend()
plt.show()