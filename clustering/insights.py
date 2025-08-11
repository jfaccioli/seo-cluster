import pandas as pd
def score_opportunities(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    df = cluster_summary.copy()
    df["score"] = 0
    df.loc[(df["impressions"] > df["impressions"].median()) & (df["ctr"] < 0.03) & (df["position"].between(5,15)), "score"] += 2
    df.loc[(df["ctr"] > 0.08) & (df["impressions"] < df["impressions"].median()) & (df["position"].between(4,10)), "score"] += 1
    df.loc[df["queries"] < 5, "score"] -= 1
    df = df.sort_values("score", ascending=False)
    return df
