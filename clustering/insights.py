import pandas as pd

def score_opportunities(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    df = cluster_summary.copy()
    df["score"] = 0
    df.loc[(df["impressions"] > df["impressions"].median()) & (df["ctr"] < 0.03) & (df["position"].between(5,15)), "score"] += 2
    df.loc[(df["ctr"] > 0.08) & (df["impressions"] < df["impressions"].median()) & (df["position"].between(4,10)), "score"] += 1
    df.loc[df["queries"] < 5, "score"] -= 1
    return df.sort_values("score", ascending=False)

def cluster_time_series(df_with_clusters: pd.DataFrame, metric: str = "Impressions") -> pd.DataFrame | None:
    """
    Returns one row per cluster with a 'trend' list for sparkline charts.
    metric: 'Impressions' or 'Clicks'
    """
    if "Date" not in df_with_clusters.columns:
        return None

    ts = df_with_clusters.dropna(subset=["Date"]).copy()
    ts["Date"] = pd.to_datetime(ts["Date"], errors="coerce")
    ts = ts.dropna(subset=["Date"])
    if ts.empty:
        return None

    # Aggregate daily per cluster
    daily = (
        ts.groupby(["Date", "cluster_id", "cluster_label"], dropna=False)
          .agg(val=(metric, "sum"))
          .reset_index()
    )

    # Ensure continuous weekly index per cluster for smoother sparkline
    out_rows = []
    for (cid, clabel), sub in daily.groupby(["cluster_id", "cluster_label"], dropna=False):
        sub = sub.set_index("Date").sort_index()

        # Weekly (Mon) resample; fallback to daily if very short
        rule = "W-MON" if (sub.index.max() - sub.index.min()).days >= 21 else "D"
        wk = sub["val"].resample(rule).sum().fillna(0)

        out_rows.append({
            "cluster_id": cid,
            "cluster_label": clabel,
            "trend": wk.tolist(),                 # list of numbers for Streamlit sparkline
            "trend_index": [d.strftime("%Y-%m-%d") for d in wk.index]  # optional
        })
    return pd.DataFrame(out_rows)
