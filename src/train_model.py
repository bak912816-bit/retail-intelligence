
import duckdb
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

# 1. Chemins
DB_PATH = "data/processed/retail.duckdb"
MODEL_PATH = Path("models/retail_sales_model.joblib")

# 2. Connexion à la base
con = duckdb.connect(DB_PATH, read_only=True)

# 3. Variables utilisées par le modèle
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

categorical_columns = ["family", "store_type"]
all_features = feature_columns + categorical_columns

# 4. Charger les données historiques
train_df = con.execute("""
    SELECT *
    FROM features.sales_features
    WHERE date <= DATE '2017-08-15'
      AND lag_14 IS NOT NULL
      AND lag_21 IS NOT NULL
      AND lag_28 IS NOT NULL
""").df()

con.close()

if train_df.empty:
    raise ValueError("Aucune donnée disponible pour entraîner le modèle.")

print("Nombre de lignes d'entraînement :", len(train_df))

# 5. Préparer les variables
X_train = pd.get_dummies(
    train_df[all_features],
    columns=categorical_columns
)

y_train = train_df["sales"]

# 6. Entraîner le modèle final
model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42
)

print("Entraînement du modèle final...")
model.fit(X_train, y_train)

# 7. Créer le dossier models s'il n'existe pas
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# 8. Sauvegarder le modèle et ses colonnes
artifact = {
    "model": model,
    "columns": X_train.columns.tolist(),
    "feature_columns": feature_columns,
    "categorical_columns": categorical_columns,
    "training_cutoff": "2017-08-15",
}

joblib.dump(artifact, MODEL_PATH)

print("\nModèle sauvegardé :", MODEL_PATH)
print("Nombre de variables après encodage :", len(X_train.columns))
print("Date limite des données d'entraînement : 2017-08-15")
