import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

# Local modules
from clustering.preprocess import normalize_text
from clustering.embed import embed_queries
from clustering.cluster import cluster_embeddings, to_umap
from clustering.label import label_clusters
from clustering.insights import score_opportunities
from utils.export import export_csv

st.set_page_config(page_title="SEO Keyword Clusters (MVP)", layout="wide")

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for target in ["query","clicks","impressions","ctr","position","date"]:
        for k,v in cols.items():
            if k == target:
                rename[v] = target.capitalize() if target == "query" else target.title()
    df = df.rename(columns=rename)
    if "Query" not in df.columns:
    # Handle 'Top queries' from some GSC exports
        if "Top queries" in df.columns:
            df = df.rename(columns={"Top queries": "Query"})
        else:
            raise ValueError("CSV must include a 'Query' column (or 'Top queries') from GSC export.")

    for c in ["Clicks","Impressions","Ctr","Position"]:
        if c not in df.columns:
            df[c] = 0.0
    return df

st.title("🔎 SEO Keyword Clusters — Interactive MVP")
st.caption("Upload a Google Search Console **Queries** CSV. Everything runs in-session; no data stored.")

with st.sidebar:
    st.header("Settings")
    min_impr = st.number_input("Min Impressions", min_value=0, value=50, step=10)
    min_cluster_size = st.number_input("Min Cluster Size (HDBSCAN)", min_value=5, value=8, step=1)
    brand_terms = st.text_input("Brand terms (comma-separated, optional)", value="")
    do_norm = st.checkbox("Normalize queries (lowercase & trim)", value=True)
    show_umap = st.checkbox("Show 2D Map (slower)", value=False)
    st.markdown("---")
    st.caption("Tip: raise Min Impressions for big CSVs (>20k rows).")

uploaded = st.file_uploader("Upload GSC Queries CSV", type=["csv"])

if uploaded is not None:
    try:
        raw = load_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    df = raw.copy()
    df = df[df["Impressions"].fillna(0) >= min_impr].reset_index(drop=True)
    df["Query_norm"] = df["Query"].astype(str).map(normalize_text) if do_norm else df["Query"].astype(str)

    if brand_terms.strip():
        brands = [b.strip().lower() for b in brand_terms.split(",") if b.strip()]
        pattern = "|".join([b.replace(" ", r"\s+") for b in brands])
        mask = ~df["Query_norm"].str.lower().str.contains(pattern, regex=True)
        df_nb = df[mask].copy()
    else:
        df_nb = df.copy()

    with st.spinner("Embedding queries…"):
        embeddings = embed_queries(df_nb["Query_norm"].tolist())

    with st.spinner("Clustering with HDBSCAN…"):
        cl_labels, probabilities = cluster_embeddings(embeddings, min_cluster_size=int(min_cluster_size))
    df_nb["cluster_id"] = cl_labels
    df_nb["cluster_prob"] = probabilities

    with st.spinner("Naming clusters…"):
        labels_df = label_clusters(df_nb, text_col="Query_norm", cluster_col="cluster_id")
    df_nb = df_nb.merge(labels_df, on="cluster_id", how="left")

    clusters = df_nb.groupby(["cluster_id","cluster_label"], dropna=False).agg(
        queries=("Query", "count"),
        clicks=("Clicks", "sum"),
        impressions=("Impressions", "sum"),
        ctr=("Ctr", "mean"),
        position=("Position", "mean"),
    ).reset_index().sort_values(["impressions","clicks"], ascending=[False, False])
    opp = score_opportunities(clusters)

    st.subheader("📊 Clusters")
    st.dataframe(clusters)

    st.subheader("💡 Opportunities")
    st.dataframe(opp)

    if show_umap:
        with st.spinner("Building 2D map…"):
            try:
                import plotly.express as px
                umap_2d = to_umap(embeddings)
                plot_df = pd.DataFrame(umap_2d, columns=["x","y"])
                plot_df["cluster_id"] = cl_labels
                plot_df["query"] = df_nb["Query"].values
                fig = px.scatter(plot_df, x="x", y="y", color=plot_df["cluster_id"].astype(str),
                                 hover_data=["query"], title="Query Map (UMAP)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render UMAP plot: {e}")

    st.download_button("⬇️ Export clusters (CSV)", data=export_csv(clusters),
                       file_name="clusters.csv", mime="text/csv")
else:
    st.info("Upload a GSC Queries CSV to get started.")
