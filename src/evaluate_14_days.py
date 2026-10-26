
import duckdb
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# 1. Paramètres
DB_PATH = "data/processed/retail.duckdb"
FORECAST_ORIGIN = "2017-08-01"
TEST_START = "2017-08-02"
TEST_END = "2017-08-15"

con = duckdb.connect(DB_PATH, read_only=True)

# 2. Variables explicatives utilisées par le modèle
feature_columns = [
    "store_nbr",
    "onpromotion",
    "store_cluster",
    "is_holiday",
    "dow",
    "dom",
    "month",
    "lag_14",
    "lag_21",
    "lag_28",
    "roll_mean_7",
    "roll_mean_27",
    "roll_std_27",
    "promo_mean_7",
]

# 3. Données d'entraînement : ventes connues jusqu'au 31 juillet
train_df = con.execute("""
    SELECT *
    FROM features.sales_features
    WHERE date <= ?
      AND lag_14 IS NOT NULL
      AND lag_21 IS NOT NULL
      AND lag_28 IS NOT NULL
    USING SAMPLE 200000 ROWS
""", [FORECAST_ORIGIN]).df()

# 4. Données à prédire : du 2 au 15 août
test_df = con.execute("""
    SELECT *
    FROM features.sales_features
    WHERE date BETWEEN ? AND ?
      AND lag_14 IS NOT NULL
      AND lag_21 IS NOT NULL
      AND lag_28 IS NOT NULL
    ORDER BY date, store_nbr, family
""", [TEST_START, TEST_END]).df()

if train_df.empty or test_df.empty:
    con.close()
    raise ValueError("Les données d'entraînement ou de test sont vides.")

print("Données d'entraînement :", len(train_df))
print("Données à prédire :", len(test_df))
print(
    "Période de prédiction :",
    test_df["date"].min().date(),
    "au",
    test_df["date"].max().date()
)

# 5. Préparer les variables
categorical_columns = ["family", "store_type"]
all_features = feature_columns + categorical_columns

X_train = pd.get_dummies(
    train_df[all_features],
    columns=categorical_columns
)

X_test = pd.get_dummies(
    test_df[all_features],
    columns=categorical_columns
)

# Aligner les colonnes entre entraînement et test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

y_train = train_df["sales"]
y_test = test_df["sales"]

# 6. Entraîner le modèle
model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42
)

print("\nEntraînement du modèle...")
model.fit(X_train, y_train)

# 7. Prédire les ventes du 2 au 15 août
predictions = model.predict(X_test)

# Les ventes ne peuvent pas être négatives
predictions = predictions.clip(min=0)

# 8. Comparer au modèle de référence : lag_14
mae_model = mean_absolute_error(y_test, predictions)
mae_baseline = mean_absolute_error(y_test, test_df["lag_14"])

# WAPE : erreur absolue totale / ventes réelles totales
wape_model = (
    abs(y_test - predictions).sum() / abs(y_test).sum()
) * 100

wape_baseline = (
    abs(y_test - test_df["lag_14"]).sum() / abs(y_test).sum()
) * 100

print("\n--- RÉSULTATS SUR 14 JOURS ---")
print("MAE du modèle :", round(mae_model, 2))
print("MAE de référence :", round(mae_baseline, 2))
print("WAPE du modèle :", round(wape_model, 2), "%")
print("WAPE de référence :", round(wape_baseline, 2), "%")

amelioration = (
    (mae_baseline - mae_model) / mae_baseline
) * 100

print("Amélioration de la MAE :", round(amelioration, 2), "%")

# 9. Sauvegarder les prédictions pour les analyser
results = test_df[["date", "store_nbr", "family", "sales"]].copy()
results = results.rename(columns={"sales": "ventes_reelles"})
results["ventes_predites"] = predictions
results["erreur_absolue"] = abs(
    results["ventes_reelles"] - results["ventes_predites"]
)

results.to_csv(
    "data/processed/predictions_14_days.csv",
    index=False
)

print("\nExemple de prédictions :")
print(results.head(10).to_string(index=False))
print("\nFichier créé : data/processed/predictions_14_days.csv")

con.close()
