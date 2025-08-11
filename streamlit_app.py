import re
import streamlit as st
import pandas as pd
import numpy as np
from typing import List

# Local modules
from clustering.preprocess import normalize_text
from clustering.embed import embed_queries
from clustering.cluster import cluster_embeddings, to_umap
from clustering.label import label_clusters
from clustering.insights import score_opportunities, cluster_time_series
from utils.export import export_csv

st.set_page_config(page_title="SEO Keyword Clusters (MVP)", layout="wide")


# --------------------------
# Robust CSV loader for GSC
# --------------------------
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    # 1) Normalize column names (case/spacing); support "Top queries"
    cols = {c.strip().lower(): c for c in df.columns}

    def find(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    q_col = find("query", "top queries")
    clk   = find("clicks")
    impr  = find("impressions")
    ctr   = find("ctr")
    pos   = find("position")
    date  = find("date")

    if not q_col:
        raise ValueError("CSV must include a 'Query' or 'Top queries' column from GSC export.")

    # 2) Rename to a stable schema
    rename = {q_col: "Query"}
    if clk:  rename[clk]  = "Clicks"
    if impr: rename[impr] = "Impressions"
    if ctr:  rename[ctr]  = "CTR"
    if pos:  rename[pos]  = "Position"
    if date: rename[date] = "Date"
    df = df.rename(columns=rename)

    # 3) Parse Date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # 4) Ensure numeric types (CTR can be '3.2%' or '0.032'; numbers can have commas)
    if "CTR" in df.columns:
        df["CTR"] = (
            df["CTR"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        ctr_numeric = pd.to_numeric(df["CTR"], errors="coerce")
        # If CTR looked like '3.2', treat as percent (3.2% -> 0.032). If already 0.032, it stays.
        df["CTR"] = np.where(ctr_numeric > 1, ctr_numeric / 100.0, ctr_numeric)

    for c in ["Clicks", "Impressions", "Position"]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )

    # 5) Fill safe defaults if any columns are missing
    for c, default in [("Clicks", 0), ("Impressions", 0), ("CTR", 0.0), ("Position", np.nan)]:
        if c not in df.columns:
            df[c] = default

    return df


# --------------------------
# UI
# --------------------------
st.title("🔎 SEO Keyword Clusters — Interactive MVP")
st.caption("Upload a Google Search Console **Queries** CSV. Everything runs in-session; no data stored.")

with st.sidebar:
    st.header("Settings")
    min_impr = st.number_input("Min Impressions", min_value=0, value=50, step=10)
    min_cluster_size = st.number_input("Min Cluster Size (HDBSCAN)", min_value=5, value=8, step=1)
    brand_terms = st.text_input("Brand terms (comma-separated, optional)", value="")
    do_norm = st.checkbox("Normalize queries (lowercase & trim)", value=True)
    show_umap = st.checkbox("Show 2D Map (slower)", value=False)
    trend_metric = st.selectbox("Trend metric", options=["Impressions", "Clicks"], index=0)
    st.markdown("---")
    st.caption("Tip: raise Min Impressions for big CSVs (>20k rows).")

uploaded = st.file_uploader("Upload GSC Queries CSV", type=["csv"])

if uploaded is not None:
    # -------- Load & prep
    try:
        raw = load_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    df = raw.copy()
    df = df[df["Impressions"].fillna(0) >= min_impr].reset_index(drop=True)
    df["Query_norm"] = df["Query"].astype(str).map(normalize_text) if do_norm else df["Query"].astype(str)

    # Brand exclusions
    if brand_terms.strip():
        brands = [b.strip().lower() for b in brand_terms.split(",") if b.strip()]
        pattern = "|".join([re.escape(b) for b in brands])
        mask = ~df["Query_norm"].str.lower().str.contains(pattern, regex=True)
        df_nb = df[mask].copy()
    else:
        df_nb = df.copy()

    # -------- Embed & cluster
    with st.spinner("Embedding queries…"):
        embeddings = embed_queries(df_nb["Query_norm"].tolist())

    with st.spinner("Clustering with HDBSCAN…"):
        cl_labels, probabilities = cluster_embeddings(embeddings, min_cluster_size=int(min_cluster_size))
    df_nb["cluster_id"] = cl_labels
    df_nb["cluster_prob"] = probabilities

    # -------- Label clusters
    with st.spinner("Naming clusters…"):
        labels_df = label_clusters(df_nb, text_col="Query_norm", cluster_col="cluster_id")
    df_nb = df_nb.merge(labels_df, on="cluster_id", how="left")

    # -------- Aggregate summary
    clusters = (
        df_nb.groupby(["cluster_id", "cluster_label"], dropna=False)
        .agg(
            queries=("Query", "count"),
            clicks=("Clicks", "sum"),
            impressions=("Impressions", "sum"),
            ctr=("CTR", "mean"),
            position=("Position", "mean"),
        )
        .reset_index()
        .sort_values(["impressions", "clicks"], ascending=[False, False])
    )

    # -------- Trend sparkline (if Date available)
    trend_df = cluster_time_series(df_nb, metric=trend_metric)
    if trend_df is not None and not trend_df.empty:
        clusters = clusters.merge(
            trend_df[["cluster_id", "cluster_label", "trend"]],
            on=["cluster_id", "cluster_label"], how="left"
        )
        st.subheader("📊 Clusters")
        st.dataframe(
            clusters,
            use_container_width=True,
            column_config={
                "trend": st.column_config.LineChartColumn(
                    "Trend",
                    help=f"Weekly {trend_metric.lower()} per cluster (sparkline)",
                    y_min=0,
                )
            }
        )
    else:
        if "Date" not in df_nb.columns or df_nb["Date"].isna().all():
            st.info("No **Date** column found, so trends are disabled. Export a GSC report with both **Date** and **Query** dimensions to see cluster trends.")
        st.subheader("📊 Clusters")
        st.dataframe(clusters, use_container_width=True)

    # -------- Opportunities
    opp = score_opportunities(clusters)
    st.subheader("💡 Opportunities")
    st.dataframe(opp, use_container_width=True)

    # -------- Optional map
    if show_umap:
        with st.spinner("Building 2D map…"):
            try:
                import plotly.express as px
                umap_2d = to_umap(embeddings)
                plot_df = pd.DataFrame(umap_2d, columns=["x", "y"])
                plot_df["cluster_id"] = cl_labels
                plot_df["query"] = df_nb["Query"].values
                fig = px.scatter(
                    plot_df, x="x", y="y",
                    color=plot_df["cluster_id"].astype(str),
                    hover_data=["query"], title="Query Map (UMAP)"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render UMAP plot: {e}")

    # -------- Export
    st.download_button(
        "⬇️ Export clusters (CSV)",
        data=export_csv(clusters),
        file_name="clusters.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a GSC Queries CSV to get started.")
