
import duckdb
import pandas as pd
import joblib

DB_PATH = "data/processed/retail.duckdb"
MODEL_PATH = "models/retail_sales_model.joblib"

# 1. Charger le modèle sauvegardé
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
model_columns = artifact["columns"]
feature_columns = artifact["feature_columns"]
categorical_columns = artifact["categorical_columns"]

print("Modèle chargé avec succès.")
print("Nombre de colonnes attendues :", len(model_columns))

# 2. Charger une période pour vérifier les prédictions
con = duckdb.connect(DB_PATH, read_only=True)

df = con.execute("""
    SELECT *
    FROM features.sales_features
    WHERE date BETWEEN DATE '2017-08-02' AND DATE '2017-08-15'
      AND lag_14 IS NOT NULL
      AND lag_21 IS NOT NULL
      AND lag_28 IS NOT NULL
    ORDER BY date, store_nbr, family
""").df()

con.close()

if df.empty:
    raise ValueError("Aucune donnée trouvée pour la période demandée.")

# 3. Préparer les variables exactement comme pendant l'entraînement
all_features = feature_columns + categorical_columns

X = pd.get_dummies(
    df[all_features],
    columns=categorical_columns
)

X = X.reindex(columns=model_columns, fill_value=0)

# 4. Prédire
df["ventes_predites"] = model.predict(X).clip(min=0)

# 5. Afficher les résultats
print("\nExemple de prédictions :")
print(
    df[["date", "store_nbr", "family", "ventes_predites"]]
    .head(10)
    .to_string(index=False)
)

# 6. Enregistrer les prédictions
output = "data/processed/predictions_model.csv"

df[["date", "store_nbr", "family", "ventes_predites"]].to_csv(
    output,
    index=False
)

print("\nPrédictions enregistrées dans :", output)
