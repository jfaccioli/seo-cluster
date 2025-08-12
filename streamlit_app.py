import re
import streamlit as st
import pandas as pd
import numpy as np
from typing import List

# Try to import weasyprint; handle if missing
try:
    from weasyprint import HTML
    import io
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

# Local modules
from clustering.preprocess import normalize_text
from clustering.embed import embed_queries
from clustering.cluster import cluster_embeddings
from clustering.label import label_clusters
from clustering.insights import score_opportunities, cluster_time_series
from clustering.brief import build_content_brief
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
# Helpers
# --------------------------
INTENT_RE = re.compile(r"^(who|what|when|where|why|how|can|does|do|is|are|should)\b", re.I)
def intent_bucket(q: str) -> str:
    s = (q or "").lower()
    if INTENT_RE.search(s) or s.endswith("?"):
        return "FAQ"
    if "near me" in s or any(x in s for x in ["perth", "wa", "closest", "local"]):
        return "Local / Navigational"
    if any(x in s for x in ["price", "cost", "quote", "book", "apply", "eligibility", "provider", "service"]):
        return "Transactional / Service"
    if any(x in s for x in ["compare", "vs", "best", "top", "reviews"]):
        return "Comparative"
    return "Informational"

def kpi_card(label: str, value: str):
    st.metric(label=label, value=value)


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
    trend_metric = st.selectbox("Trend metric", options=["Impressions", "Clicks"], index=0)
    st.markdown("---")
    st.caption("Tip: raise Min Impressions for big CSVs (>20k rows).")
    st.caption("Use 'Re-cluster Unclustered' button if unclustered share is high (>50%) to find finer groups.")
    if WEASYPRINT_AVAILABLE:
        st.caption("Download PDF reports for professional summaries under Content Brief.")
    else:
        st.caption("PDF export unavailable (requires weasyprint library).")

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

    # Important: reindex after brand filter so embeddings align with row positions
    df_nb = df_nb.reset_index(drop=True)

    # -------- Embed & cluster
    with st.spinner("Embedding queries…"):
        embeddings = embed_queries(df_nb["Query_norm"].tolist())

    with st.spinner("Clustering with HDBSCAN…"):
        cl_labels, probabilities = cluster_embeddings(embeddings, min_cluster_size=int(min_cluster_size))
    df_nb["cluster_id"] = cl_labels
    df_nb["cluster_prob"] = probabilities

    # -------- Label clusters (impressions-weighted, smarter stopwords)
    with st.spinner("Naming clusters…"):
        labels_df = label_clusters(
            df_nb,
            text_col="Query_norm",
            cluster_col="cluster_id",
            weight_col="Impressions"
        )
    df_nb = df_nb.merge(labels_df, on="cluster_id", how="left")

    # Mark unclustered nicely
    df_nb["cluster_label"] = np.where(
        df_nb["cluster_id"] == -1,
        "Unclustered (miscellaneous)",
        df_nb["cluster_label"].fillna("")
    )

    # Re-cluster unclustered button
    if st.button("Re-cluster Unclustered Queries (with smaller min size)"):
        unclustered = df_nb[df_nb["cluster_id"] == -1].copy()
        if not unclustered.empty:
            with st.spinner("Re-clustering unclustered queries…"):
                embeddings_uncl = embed_queries(unclustered["Query_norm"].tolist())
                cl_labels_uncl, probs_uncl = cluster_embeddings(embeddings_uncl, min_cluster_size=3)
                unclustered["cluster_id"] = cl_labels_uncl
                unclustered["cluster_prob"] = probs_uncl
                labels_uncl = label_clusters(
                    unclustered,
                    text_col="Query_norm",
                    cluster_col="cluster_id",
                    weight_col="Impressions"
                )
                unclustered = unclustered.merge(labels_uncl, on="cluster_id", how="left")
                # Ensure cluster_label exists and handle NaNs
                if "cluster_label" not in unclustered.columns:
                    unclustered["cluster_label"] = ""
                unclustered["cluster_label"] = unclustered["cluster_label"].fillna("")
                unclustered["cluster_label"] = np.where(
                    unclustered["cluster_id"] == -1,
                    "Remaining Unclustered",
                    unclustered["cluster_label"]
                )
                # Update original df_nb with re-clustered rows
                df_nb.update(unclustered)
                # Count re-clustered vs remaining
                new_clusters = unclustered[unclustered["cluster_id"] != -1]["cluster_id"].nunique()
                remaining_unclustered = len(unclustered[unclustered["cluster_id"] == -1])
                total_reclustered = len(unclustered)
                st.success(
                    f"Re-clustered {total_reclustered} queries: {new_clusters} new clusters formed, "
                    f"{remaining_unclustered} remain unclustered. Refresh visuals below."
                )

    # Intent tagging for dashboard breakdowns
    df_nb["intent"] = df_nb["Query_norm"].map(intent_bucket)

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
    show_trend = trend_df is not None and not trend_df.empty
    if show_trend:
        clusters = clusters.merge(
            trend_df[["cluster_id", "cluster_label", "trend"]],
            on=["cluster_id", "cluster_label"], how="left"
        )

    # -------- Opportunities
    opp = score_opportunities(clusters)  # columns: cluster_id, queries, clicks, impressions, ctr, position, score

    # ==========================
    # DASHBOARD
    # ==========================
    st.header("📊 Dashboard")

    # KPI row
    total_impr = float(df_nb["Impressions"].sum() or 0)
    unclustered_impr = float(df_nb.loc[df_nb["cluster_id"] == -1, "Impressions"].sum() or 0)
    clustered_impr = max(total_impr - unclustered_impr, 0)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1: kpi_card("Impressions", f"{int(total_impr):,}")
    with col2: kpi_card("Clicks", f"{int(df_nb['Clicks'].sum()):,}")
    with col3: kpi_card("Avg CTR", f"{(df_nb['CTR'].mean()*100 if df_nb['CTR'].notna().any() else 0):.2f}%")
    with col4: kpi_card("Avg Position", f"{df_nb['Position'].mean():.2f}" if df_nb["Position"].notna().any() else "—")
    with col5: kpi_card("#
