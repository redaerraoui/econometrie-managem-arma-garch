"""Étude économétrique du cours mensuel de Managem (août 2000 – janvier 2026).

Reproduit en Python les principaux résultats du rapport (estimés dans EViews) :
statistiques descriptives, autocorrélations, tests ADF et Phillips-Perron, ARMA(1,1),
GARCH / GJR (TARCH) / EGARCH, puis compare les prévisions hors échantillon (découpage 80/20).

    python analyse.py
"""
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from arch.unitroot import PhillipsPerron
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf

warnings.filterwarnings("ignore")


def to_float(s):
    """Convertit le format Investing.com '1.149,86' en nombre."""
    return s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float)


# 1. Données
df = pd.read_csv("data/managem_mensuel.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["Cours"] = to_float(df["Dernier"])
df = df.sort_values("Date").reset_index(drop=True)
P = df["Cours"]
L = np.log(P)
R = L.diff().dropna().reset_index(drop=True)          # rendements logarithmiques mensuels

print(f"{len(P)} observations, du {df.Date.iloc[0]:%m/%Y} au {df.Date.iloc[-1]:%m/%Y}")
print(f"Cours : moyenne {P.mean():.2f}, écart-type {P.std():.2f}, min {P.min():.0f}, max {P.max():.0f}")

# 2. Autocorrélations du cours en niveau
a, pa = acf(P, nlags=24), pacf(P, nlags=2)
print(f"ACF du cours : AC1 = {a[1]:.3f}, AC12 = {a[12]:.3f}, AC24 = {a[24]:.3f} ; PAC2 = {pa[2]:.3f}")

# 3. Racine unitaire (retard choisi par le critère de Schwarz, 15 retards au plus, comme EViews)
for nom, serie, reg in [("Cours (constante + tendance)", P, "ct"), ("Log du cours (constante)", L, "c"),
                        ("Rendements (constante)", R, "c")]:
    stat, p, lag, *_ = adfuller(serie, maxlag=15, regression=reg, autolag="BIC")
    print(f"ADF {nom:30s} stat = {stat:8.4f}  p = {p:.4f}  retards = {lag}")
for nom, serie, tr in [("Cours", P, "ct"), ("Rendements", R, "c")]:
    pp = PhillipsPerron(serie, trend=tr)
    print(f"Phillips-Perron {nom:12s} stat = {pp.stat:8.4f}  p = {pp.pvalue:.4f}")

# 4. Moyenne conditionnelle : ARMA(1,1)
arma = ARIMA(R.values, order=(1, 0, 1)).fit()
print("ARMA(1,1) :", dict(zip(["c", "AR(1)", "MA(1)", "sigma2"], arma.params.round(4))))

# 5. Volatilité conditionnelle (moyenne constante ; EViews estime ARMA et variance conjointement)
for nom, kw in [("GARCH(1,1)", dict(p=1, q=1)), ("GJR / TARCH", dict(p=1, o=1, q=1)),
                ("EGARCH(1,1)", dict(vol="EGARCH", p=1, o=1, q=1))]:
    res = arch_model(R.values, mean="Constant", dist="normal", **kw).fit(disp="off")
    print(f"{nom:12s}", res.params.round(4).to_dict())

# 6. Prévision hors échantillon : 80 % pour l'estimation, 20 % pour le test
n = len(R)
n_train = int(0.8 * n)
r = R.values
test = r[n_train:]


def scores(pred):
    e = test - pred
    return np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e))


fit = ARIMA(r[:n_train], order=(1, 0, 1)).fit()
results = {
    "Rendement nul": scores(np.zeros_like(test)),
    "Moyenne d'estimation": scores(np.full_like(test, r[:n_train].mean())),
    "ARMA(1,1) dynamique": scores(fit.forecast(len(test))),
    "ARMA(1,1) à un pas": scores(fit.apply(r).predict()[n_train:]),
}

# MLP 2-10-1 : entrées R_{t-1}, R_{t-2}, cible R_t (moyenne sur 10 initialisations)
X = np.column_stack([r[1:-1], r[:-2]])
y = r[2:]
k = n_train - 2


def mlp_scores(inputs):
    out = []
    for seed in range(10):
        m = MLPRegressor(hidden_layer_sizes=(10,), activation="relu", solver="adam",
                         max_iter=2000, random_state=seed).fit(inputs[:k], y[:k])
        out.append(scores(m.predict(inputs[k:])))
    return tuple(np.mean(out, axis=0))


results["MLP (R_t-1, R_t-2)"] = mlp_scores(X)
# Contrôle : si R_t se retrouve parmi les entrées (fuite d'information), l'erreur chute artificiellement
results["MLP avec R_t en entrée (fuite)"] = mlp_scores(np.column_stack([r[2:], r[1:-1]]))

print(f"\nPrévision hors échantillon ({len(test)} mois)")
for nom, (rmse, mae) in results.items():
    print(f"  {nom:32s} RMSE = {rmse:.4f}  MAE = {mae:.4f}")
