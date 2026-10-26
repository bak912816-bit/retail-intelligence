
import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Retail Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("Retail Intelligence")
st.subheader("Analyse des ventes au détail")

# Connexion à la base de données DuckDB
con = duckdb.connect("data/processed/retail.duckdb", read_only=True)

# Chargement des ventes hebdomadaires par famille
df = con.sql("""
    SELECT family, week, sales
    FROM features.sales_by_family_week
    ORDER BY week
""").df()

# Indicateurs principaux
col1, col2 = st.columns(2)

with col1:
    st.metric("Ventes totales", f"{df['sales'].sum():,.0f}")

with col2:
    st.metric("Familles de produits", df["family"].nunique())

# Graphique interactif
st.subheader("Évolution des ventes par famille")

families = sorted(df["family"].unique())
selected_families = st.multiselect(
    "Choisir les familles",
    families,
    default=families[:3]
)

filtered_df = df[df["family"].isin(selected_families)]

fig = px.line(
    filtered_df,
    x="week",
    y="sales",
    color="family",
    title="Ventes hebdomadaires"
)

st.plotly_chart(fig, use_container_width=True)

# Top 10 des familles par ventes totales
st.subheader("Top 10 des familles par ventes totales")

top_families = (
    df.groupby("family", as_index=False)["sales"]
    .sum()
    .sort_values("sales", ascending=False)
    .head(10)
)

fig_top = px.bar(
    top_families,
    x="sales",
    y="family",
    orientation="h",
    title="Les 10 familles les plus performantes",
    labels={
        "sales": "Ventes totales",
        "family": "Famille de produits"
    }
)

fig_top.update_layout(yaxis={"categoryorder": "total ascending"})

st.plotly_chart(fig_top, use_container_width=True)

# --------------------------------------------------
# PRÉVISIONS FUTURES — SCÉNARIO INDICATIF
# --------------------------------------------------

import pandas as pd
import streamlit as st
from pathlib import Path

PREDICTIONS_PATH = Path(
    "data/processed/predictions_future_14_days.csv"
)

st.header("Prévisions des ventes — 14 jours")

st.warning(
    "Scénario indicatif du 16 au 29 août 2017. "
    "Les promotions futures sont supposées nulles. "
    "Les ventes réelles de cette période ne sont pas disponibles."
)

if PREDICTIONS_PATH.exists():
    predictions = pd.read_csv(PREDICTIONS_PATH)
    predictions["date"] = pd.to_datetime(predictions["date"])

    total_forecast = predictions["ventes_predites"].sum()

    col1, col2 = st.columns(2)

    col1.metric(
        "Ventes prédites (14 jours)",
        f"{total_forecast:,.0f}"
    )

    col2.metric(
        "Dates couvertes",
        predictions["date"].nunique()
    )

    daily_sales = (
        predictions.groupby("date", as_index=False)
        ["ventes_predites"].sum()
    )

    st.subheader("Évolution quotidienne prévue")

    st.line_chart(
        daily_sales.set_index("date")["ventes_predites"]
    )

    st.subheader("Prévisions par famille de produits")

    family_sales = (
        predictions.groupby("family", as_index=False)
        ["ventes_predites"].sum()
        .sort_values("ventes_predites", ascending=False)
    )

    st.bar_chart(
        family_sales.set_index("family")["ventes_predites"]
    )

    st.subheader("Détail des prévisions")
    st.dataframe(predictions, use_container_width=True)

    st.download_button(
        "Télécharger les prévisions CSV",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name="predictions_future_14_days.csv",
        mime="text/csv"
    )

else:
    st.error(
        "Fichier de prévisions introuvable. "
        "Exécute d'abord : python src/predict_next_14_days.py"
    )


con.close()
