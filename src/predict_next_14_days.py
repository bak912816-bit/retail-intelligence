
"""
PRÉVISION DES VENTES RETAIL — 14 JOURS FUTURS

OBJECTIF
Produire des prévisions pour la période du 16 au 29 août 2017.

HYPOTHÈSES ET LIMITES
1. Les ventes historiques disponibles s'arrêtent au 15 août 2017.
2. Aucune promotion future n'est connue. Nous supposons donc
   onpromotion = 0 pour toutes les prévisions.
3. Les retards de ventes et les moyennes mobiles sont calculés
   à partir des ventes historiques disponibles, sans utiliser
   les ventes réelles des dates à prédire.
4. Le calendrier des jours fériés provient de la base disponible.
5. Les résultats sont un scénario indicatif, pas une garantie
   des ventes qui se seraient réellement produites.
6. La précision de ces nouvelles prévisions ne peut pas être
   évaluée sans disposer des ventes réelles de cette période.

Le modèle utilisé a été entraîné sur les données disponibles
jusqu'au 15 août 2017.
"""

import duckdb
import pandas as pd
import joblib
from pathlib import Path

DB_PATH = "data/processed/retail.duckdb"
MODEL_PATH = Path("models/retail_sales_model.joblib")
OUTPUT_PATH = Path("data/processed/predictions_future_14_days.csv")

START_DATE = "2017-08-16"
END_DATE = "2017-08-29"

# Charger le modèle final et ses colonnes d'entraînement
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
model_columns = artifact["columns"]
feature_columns = artifact["feature_columns"]
categorical_columns = artifact["categorical_columns"]

con = duckdb.connect(DB_PATH, read_only=True)

# Construire les variables des dates futures.
# Les ventes futures sont NULL : elles ne sont jamais utilisées
# comme valeurs réelles pour calculer les variables explicatives.
future_features = con.execute("""
    WITH future_grid AS (
        SELECT
            CAST(g.date AS DATE) AS date,
            sf.store_nbr,
            sf.family,
            NULL::DOUBLE AS sales,
            0 AS onpromotion
        FROM generate_series(
            DATE '2017-08-16',
            DATE '2017-08-29',
            INTERVAL 1 DAY
        ) AS g(date)
        CROSS JOIN (
            SELECT DISTINCT store_nbr, family
            FROM clean.sales
        ) AS sf
    ),

    all_sales AS (
        SELECT
            date,
            store_nbr,
            family,
            sales,
            onpromotion
        FROM clean.sales
        WHERE date <= DATE '2017-08-15'

        UNION ALL

        SELECT
            date,
            store_nbr,
            family,
            sales,
            onpromotion
        FROM future_grid
    ),

    historical_features AS (
        SELECT
            date,
            store_nbr,
            family,
            sales,
            onpromotion,

            LAG(sales, 14) OVER p AS lag_14,
            LAG(sales, 21) OVER p AS lag_21,
            LAG(sales, 28) OVER p AS lag_28,

            AVG(sales) OVER (
                p ROWS BETWEEN 20 PRECEDING AND 14 PRECEDING
            ) AS roll_mean_7,

            AVG(sales) OVER (
                p ROWS BETWEEN 40 PRECEDING AND 14 PRECEDING
            ) AS roll_mean_27,

            STDDEV(sales) OVER (
                p ROWS BETWEEN 40 PRECEDING AND 14 PRECEDING
            ) AS roll_std_27,

            AVG(onpromotion) OVER (
                p ROWS BETWEEN 20 PRECEDING AND 14 PRECEDING
            ) AS promo_mean_7

        FROM all_sales

        WINDOW p AS (
            PARTITION BY store_nbr, family
            ORDER BY date
        )
    )

    SELECT
        f.date,
        f.store_nbr,
        f.family,
        f.onpromotion,

        st.type AS store_type,
        st.cluster AS store_cluster,

        COALESCE(h.is_holiday, 0) AS is_holiday,

        DAYOFWEEK(f.date) AS dow,
        DAY(f.date) AS dom,
        MONTH(f.date) AS month,

        f.lag_14,
        f.lag_21,
        f.lag_28,
        f.roll_mean_7,
        f.roll_mean_27,
        f.roll_std_27,
        f.promo_mean_7

    FROM historical_features f

    LEFT JOIN raw.stores st
        ON f.store_nbr = st.store_nbr

    LEFT JOIN (
        SELECT date, MAX(is_holiday) AS is_holiday
        FROM clean.national_holidays
        GROUP BY date
    ) h
        ON f.date = h.date

    WHERE f.date BETWEEN DATE '2017-08-16'
                     AND DATE '2017-08-29'

    ORDER BY f.date, f.store_nbr, f.family
""").df()

con.close()

if future_features.empty:
    raise ValueError("Aucune variable future n'a été créée.")

# Vérifier que les variables historiques sont disponibles
required_features = [
    "lag_14", "lag_21", "lag_28",
    "roll_mean_7", "roll_mean_27",
    "roll_std_27", "promo_mean_7"
]

if future_features[required_features].isna().any().any():
    raise ValueError(
        "Certaines variables historiques sont manquantes. "
        "Vérifie la construction des données futures."
    )

# Préparer les variables exactement comme lors de l'entraînement
X = future_features[feature_columns + categorical_columns].copy()

X = pd.get_dummies(
    X,
    columns=categorical_columns
)

# Garantir les mêmes colonnes et le même ordre qu'à l'entraînement
X = X.reindex(columns=model_columns, fill_value=0)

# Prédire les ventes ; les valeurs négatives sont ramenées à zéro
predictions = model.predict(X).clip(min=0)

# Construire le fichier de résultats
results = future_features[
    ["date", "store_nbr", "family"]
].copy()

results["ventes_predites"] = predictions
results["scenario_promotions"] = (
    "Aucune promotion future supposée"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

print("Prévisions futures générées avec succès.")
print(f"Période : {START_DATE} au {END_DATE}")
print(f"Nombre de prévisions : {len(results)}")
print(f"Fichier enregistré : {OUTPUT_PATH}")

print("\nExemple de prévisions :")
print(results.head(10).to_string(index=False))

print("\nVentes prédites totales sur toutes les familles et magasins :")
print(round(results["ventes_predites"].sum(), 2))
