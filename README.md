# Étude économétrique et prévision du cours de Managem

Projet de séries temporelles réalisé par **Reda Erraoui**, élève ingénieur à l'ENSA Agadir, filière Finance et Ingénierie Décisionnelle, sous l'encadrement de Mme S. Hadiri (2025–2026).

**[Essayer le simulateur en ligne](https://redaerraoui.github.io/simulateurs/arma-garch.html)** · **[Lire le rapport (PDF)](rapport.pdf)**

## Données

`data/managem_mensuel.csv` contient le cours mensuel de clôture de Managem à la Bourse de Casablanca, d'août 2000 à janvier 2026 (306 observations, format Investing.com).

## Démarche

1. **Analyse exploratoire.** Statistiques descriptives et autocorrélogrammes du cours en niveau.
2. **Stationnarité.** Tests de Dickey-Fuller augmenté et de Phillips-Perron sur le cours, son logarithme et les rendements logarithmiques.
3. **Moyenne conditionnelle.** Modèle ARMA(1,1) sur les rendements, après exclusion des modèles ARFIMA et SARIMA.
4. **Volatilité conditionnelle.** Test ARCH-LM, puis modèles GARCH(1,1), TARCH et EGARCH.
5. **Prévision et risque.** Prévision de la volatilité et Value-at-Risk conditionnelle.
6. **Sensibilité aux matières premières.** Régression du cours sur les prix de l'or, de l'argent et du cuivre en dirhams (2022–2026).
7. **Intelligence artificielle.** Comparaison avec un réseau de neurones MLP.

## Principaux résultats

| Élément | Résultat |
|---|---|
| ADF, cours en niveau (constante et tendance) | 5,88 (p = 1,00) : non stationnaire |
| ADF, rendements (constante) | −15,17 (p < 0,0001) : stationnaire |
| Ordre d'intégration du cours | I(1) |
| ARMA(1,1) | AR = 0,77 ; MA = −0,68 ; R² = 0,02 |
| GARCH(1,1), EViews | α = 0,149 ; β = 0,744 ; persistance de 0,89 (demi-vie d'environ 6 mois) |
| EGARCH, EViews | coefficient d'asymétrie de −0,133 (p = 0,0001) : effet de levier |
| Régression sur l'or (2022–2026) | β = 0,252 ; R² = 0,881 (en niveaux) |

`analyse.py` reproduit en Python les statistiques descriptives, les autocorrélations, les tests ADF et l'ARMA(1,1) du rapport, à l'arrondi près. Les modèles GARCH sont réestimés avec le package `arch`, avec une moyenne constante. Leurs coefficients diffèrent donc légèrement de ceux d'EViews, qui estime conjointement l'ARMA et la variance.

## Remarques et vérifications (octobre 2026)

- **Prévision hors échantillon.** Sur le découpage 80/20 (test de janvier 2021 à janvier 2026, 61 mois), aucun modèle ne fait nettement mieux qu'une prévision nulle :

  | Méthode | RMSE | MAE |
  |---|---:|---:|
  | Rendement nul | 0,0920 | 0,0647 |
  | ARMA(1,1), prévision dynamique | 0,0908 | 0,0641 |
  | ARMA(1,1), prévision à un pas | 0,0891 | 0,0646 |
  | MLP 2-10-1, entrées R(t−1) et R(t−2), moyenne de 10 initialisations | 0,0976 | 0,0740 |
  | MLP avec R(t) parmi les entrées (fuite d'information) | 0,0626 | 0,0464 |

  Le rapport donne un RMSE de 0,0516 pour le MLP contre 0,1131 pour l'ARMA. On ne retrouve un écart de cet ordre que si le rendement à prévoir figure parmi les entrées du réseau. L'avantage du MLP semble donc provenir d'une fuite d'information plutôt que d'une non-linéarité. Le code du MLP du rapport permettrait de le confirmer.
- **ARMA(1,1).** Les racines AR (0,77) et MA (0,68) sont proches et se compensent en grande partie. Avec un R² de 0,02, le modèle reste très proche d'un bruit blanc.
- **TARCH.** Le coefficient ARCH estimé est négatif (−0,055), ce qui sort de la contrainte de positivité habituelle. L'EGARCH, qui n'impose pas cette contrainte, confirme l'effet de levier.
- **Régression sur l'or.** Estimée en niveaux sur deux séries non stationnaires, avec une statistique de Durbin-Watson de 0,30, elle peut refléter une tendance commune. Une estimation en rendements ou un test de cointégration permettrait de trancher. Le rapport le signale déjà. Les prix des métaux ne sont pas inclus dans ce dépôt.

## Lancement

```bash
pip install -r requirements.txt
python analyse.py
```

## Contact

Reda Erraoui · [Portfolio](https://redaerraoui.github.io) · redaerraouie@gmail.com
