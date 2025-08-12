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
    clk = find("clicks")
    impr = find("impressions")
    ctr = find("ctr")
    pos = find("position")
    date = find("date")

    if not q_col:
        raise ValueError("CSV must include a 'Query' or 'Top queries' column from GSC export.")

    # 2) Rename to a stable schema
    rename = {q_col: "Query"}
    if clk: rename[clk] = "Clicks"
    if impr: rename[impr] = "Impressions"
    if ctr: rename[ctr] = "CTR"
    if pos: rename[pos] = "Position"
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

    # Debug: Log clusters shape and KPI values
    st.write(f"Debug: clusters shape: {clusters.shape}")
    st.write(f"Debug: KPI values - Impressions: {int(total_impr):,}, Clicks: {int(df_nb['Clicks'].sum()):,}, "
             f"Avg CTR: {(df_nb['CTR'].mean()*100 if df_nb['CTR'].notna().any() else 0):.2f}%, "
             f"Avg Position: {df_nb['Position'].mean():.2f if df_nb['Position'].notna().any() else '—'}, "
             f"Num Clusters: {(clusters['cluster_id'] != -1).sum():,}")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        kpi_card("Impressions", str(int(total_impr)))
    with col2:
        kpi_card("Clicks", str(int(df_nb["Clicks"].sum())))
    with col3:
        kpi_card("Avg CTR", f"{(df_nb['CTR'].mean()*100 if df_nb['CTR'].notna().any() else 0):.2f}%")
    with col4:
        kpi_card("Avg Position", f"{df_nb['Position'].mean():.2f}" if df_nb["Position"].notna().any() else "—")
    with col5:
        kpi_card("Num Clusters", str((clusters["cluster_id"] != -1).sum()))
    with col6:
        clustered_only = clusters[clusters["cluster_id"] != -1]
        top10_impr = float(clustered_only.head(10)["impressions"].sum() or 0)
        share_top10 = (top10_impr / clustered_impr) if clustered_impr > 0 else 0.0
        kpi_card("Top10 Share (clustered)", f"{share_top10*100:.1f}%")

    # Show Unclustered share
    st.caption(f"Unclustered share: {(unclustered_impr/total_impr*100 if total_impr>0 else 0):.1f}%")

    # Filters on dashboard (cluster multi-select)
    clusters["_label_for_ui"] = clusters.apply(
        lambda r: f"[{int(r['cluster_id'])}] {r['cluster_label'] or ''}".strip(),
        axis=1
    )
    sel_clusters = st.multiselect(
        "Filter clusters (optional)",
        options=clusters["_label_for_ui"].tolist(),
        default=clusters["_label_for_ui"].head(10).tolist()
    )
    def parse_id(x: str) -> int:
        try: return int(x.split("]")[0].strip("["))
        except: return -9999
    selected_ids = {parse_id(x) for x in sel_clusters} if sel_clusters else None

    # Subset for visuals
    clusters_v = clusters.copy()
    df_vis = df_nb.copy()
    if selected_ids:
        clusters_v = clusters_v[clusters_v["cluster_id"].isin(selected_ids)]
        df_vis = df_vis[df_vis["cluster_id"].isin(selected_ids)]

    import plotly.express as px

    # Viz 1: Treemap (size=Impressions, color=CTR)
    st.subheader("Treemap — Cluster scale & CTR")
    if not clusters_v.empty:
        fig_tm = px.treemap(
            clusters_v,
            path=["cluster_label"],
            values="impressions",
            color="ctr",
            color_continuous_scale="Blues",
            hover_data={"queries": True, "clicks": True, "position": True, "ctr": True}
        )
        fig_tm.update_layout(margin=dict(t=30,l=0,r=0,b=0))
        st.plotly_chart(fig_tm, use_container_width=True)
    else:
        st.info("No clusters to show. Try adjusting filters.")

    # Viz 2: Opportunity bubble (x=position, y=impressions, size=clicks, color=score)
    st.subheader("Opportunity Map — Where to act next")
    if not clusters_v.empty:
        # No need to merge—opp already has cluster_label from clusters
        opp_v = opp.copy()

        if selected_ids:
            opp_v = opp_v[opp_v["cluster_id"].isin(selected_ids)]

        # Ensure numeric types and handle missing/invalid values
        required_cols = ["position", "impressions", "clicks", "score"]
        for c in required_cols:
            if c in opp_v.columns:
                opp_v[c] = pd.to_numeric(opp_v[c], errors="coerce")
            else:
                st.error(f"Missing required column: {c}")
                st.stop()

        # Replace infinities and drop rows with NaN in required columns (add cluster_label to ensure hover_name works)
        opp_v = opp_v.replace([np.inf, -np.inf], np.nan)
        opp_v = opp_v.dropna(subset=required_cols + ["cluster_label"])

        if opp_v.empty:
            st.info("No data for the Opportunity Map with current filters.")
        else:
            fig_bub = px.scatter(
                opp_v,
                x="position",
                y="impressions",
                size="clicks",
                color="score",
                hover_name="cluster_label",
                size_max=60,
                labels={"position": "Avg Position (lower is better)", "impressions": "Impressions"}
            )
            fig_bub.update_layout(margin=dict(t=30, l=0, r=0, b=0), xaxis_autorange="reversed")
            st.plotly_chart(fig_bub, use_container_width=True)
    else:
        st.info("No opportunities to display.")

    # Viz 3: Trend line (if Date available)
    if "Date" in df_vis.columns and df_vis["Date"].notna().any():
        st.subheader(f"Trend — {trend_metric} by cluster")
        ts = (
            df_vis.dropna(subset=["Date"])
                 .groupby([pd.Grouper(key="Date", freq="W-MON"), "cluster_label"], dropna=False)
                 .agg(val=(trend_metric, "sum"))
                 .reset_index()
        )
        if not ts.empty:
            fig_line = px.line(ts, x="Date", y="val", color="cluster_label",
                               labels={"val": trend_metric})
            fig_line.update_layout(margin=dict(t=30,l=0,r=0,b=0))
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No dated data to plot trends.")
    else:
        st.info("No **Date** column found in CSV, so trend charts are hidden. Export a Date+Query GSC report to enable.")

    # Viz 4: Intent breakdown
    st.subheader("Intent breakdown")
    if not df_vis.empty:
        intent_pivot = (
            df_vis.groupby(["cluster_label","intent"])
                  .size().reset_index(name="count")
        )
        fig_intent = px.bar(intent_pivot, x="cluster_label", y="count", color="intent", barmode="stack")
        fig_intent.update_layout(margin=dict(t=30,l=0,r=0,b=0), xaxis={'visible': False, 'showticklabels': False})
        st.plotly_chart(fig_intent, use_container_width=True)

    # Viz 5: Top queries table (drilldown)
    st.subheader("Top queries (drilldown)")
    topq = (
        df_vis.sort_values(["Impressions","Clicks"], ascending=[False, False])
              [["cluster_label","Query","Clicks","Impressions","CTR","Position"]]
              .head(500)
    )
    st.dataframe(topq, use_container_width=True)

    # ==========================
    # CLUSTERS + OPPORTUNITIES (tables)
    # ==========================
    st.header("📋 Tables")

    # Clusters table (with or without trend)
    st.subheader("Clusters")
    if trend_df is not None and not trend_df.empty:
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
        st.dataframe(clusters, use_container_width=True)

    st.subheader("Opportunities")
    st.dataframe(opp, use_container_width=True)

    # ==========================
    # CONTENT BRIEF
    # ==========================
    st.header("📄 Content Brief")
    if clusters.empty:
        st.info("No clusters yet. Upload a CSV or adjust filters.")
    else:
        # Cluster selector label
        clusters["_label_for_ui2"] = clusters.apply(
            lambda r: f"[{int(r['cluster_id'])}] {r['cluster_label'] or ''}".strip(),
            axis=1
        )
        sel = st.selectbox("Choose a cluster for a brief", options=clusters["_label_for_ui2"].tolist())
        chosen_id = int(sel.split("]")[0].strip("[")) if sel else None
        chosen_row = clusters[clusters["cluster_id"] == chosen_id].head(1)
        chosen_label = chosen_row["cluster_label"].iloc[0] if not chosen_row.empty else ""
        # Ensure chosen_label is a valid string
        if pd.isna(chosen_label) or not isinstance(chosen_label, str) or not chosen_label.strip():
            st.warning(f"Invalid cluster label for ID {chosen_id}. Using fallback label.")
            chosen_label = f"Cluster {chosen_id}"

        brief_md = ""
        if chosen_id == -1:
            st.info("This is the **Unclustered** group. It contains mixed queries, so a single content brief isn’t useful. Adjust filters or clustering settings to reduce noise.")
        else:
            df_cluster = df_nb[df_nb["cluster_id"] == chosen_id].copy()
            try:
                brief_md = build_content_brief(
                    df_cluster=df_cluster,
                    cluster_id=chosen_id,
                    cluster_label=chosen_label,
                    centroids=None  # optional
                )
                st.markdown(brief_md)
                st.download_button(
                    "⬇️ Download brief (Markdown)",
                    data=brief_md.encode("utf-8"),
                    file_name=f"content-brief-cluster-{chosen_id}.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"Failed to generate content brief: {str(e)}")

        # PDF Export
        if WEASYPRINT_AVAILABLE:
            if st.button("⬇️ Download PDF Report"):
                try:
                    html_content = f"""
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #1f77b4; }}
                        h2 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                    <h1>SEO Cluster Report</h1>
                    <h2>Summary</h2>
                    <p><strong>Impressions:</strong> {int(total_impr):,}</p>
                    <p><strong>Clicks:</strong> {int(df_nb['Clicks'].sum()):,}</p>
                    <p><strong>Avg CTR:</strong> {(df_nb['CTR'].mean()*100 if df_nb['CTR'].notna().any() else 0):.2f}%</p>
                    <p><strong>Avg Position:</strong> {df_nb['Position'].mean():.2f if df_nb['Position'].notna().any() else '—'}</p>
                    <p><strong>Clusters:</strong> {(clusters['cluster_id'] != -1).sum():,}</p>
                    <p><strong>Top10 Share (clustered):</strong> {share_top10*100:.1f}%</p>
                    <p><strong>Unclustered Share:</strong> {(unclustered_impr/total_impr*100 if total_impr>0 else 0):.1f}%</p>
                    <h2>Clusters</h2>
                    {clusters.to_html(index=False)}
                    <h2>Selected Content Brief</h2>
                    {brief_md if brief_md else '<p>No brief selected or available.</p>'}
                    """
                    pdf_buffer = io.BytesIO()
                    HTML(string=html_content).write_pdf(pdf_buffer)
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name="seo_cluster_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Failed to generate PDF report: {str(e)}")
        else:
            st.info("PDF export requires the 'weasyprint' library. Install it to enable this feature.")

    # Export summary
    st.download_button(
        "⬇️ Export clusters (CSV)",
        data=export_csv(clusters),
        file_name="clusters.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a GSC Queries CSV to get started.")
