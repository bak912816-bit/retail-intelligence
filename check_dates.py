import duckdb

con = duckdb.connect(
    "data/processed/retail.duckdb",
    read_only=True
)

print(con.execute("""
    SELECT
        COUNT(*) AS nombre_lignes,
        COUNT(DISTINCT (date, store_nbr, family)) AS combinaisons_uniques,
        COUNT(DISTINCT store_nbr) AS magasins,
        COUNT(DISTINCT family) AS familles,
        COUNT(DISTINCT date) AS dates
    FROM clean.sales
""").fetchdf().to_string(index=False))

con.close()