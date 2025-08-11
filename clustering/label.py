# clustering/label.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Extra generic stopwords common in GSC queries that hurt labels
GENERIC_STOPWORDS = {
    "near", "nearby", "in", "for", "to", "from", "and", "or", "with", "without",
    "service", "services", "best", "top", "review", "reviews", "clinic", "provider",
    "providers", "perth", "wa", "australia", "cost", "price", "prices", "pricing",
    "book", "booking", "open", "opening", "hours", "today", "now", "my", "your",
    "find", "get", "list", "local", "area", "around", "near me"
}

def _build_stopwords(user_stopwords=None):
    sw = set(GENERIC_STOPWORDS)
    if user_stopwords:
        sw |= set(user_stopwords)
    return sw

def _c_tfidf(docs: list[str], weights: np.ndarray | None, stop_words_list) -> tuple[np.ndarray, np.ndarray]:
    """
    Lightweight c-TF-IDF:
    - Build CountVectorizer over docs (each doc = one cluster concatenated)
    - Compute class-based tf * idf
    - Optionally weight term counts per doc by provided weights (e.g., impressions)
    Returns: (scores_matrix, terms)
    """
    if not docs:
        return np.zeros((0, 0)), np.array([])

    # sklearn requires stop_words to be None, 'english', or an iterable (list/tuple), not a set
    cv = CountVectorizer(ngram_range=(1, 3), stop_words=stop_words_list, min_df=1)
    X = cv.fit_transform(docs)  # shape: (n_docs, n_terms)
    terms = cv.get_feature_names_out()

    tf = X.toarray().astype(float)  # term frequency per doc

    # If weights provided, scale term frequencies for each doc
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        tf = tf * (w / (w.max() if w.max() else 1.0))

    # c-TF-IDF idf: log(N / df)
    df_counts = (X > 0).sum(axis=0).A1  # document frequency per term
    N = len(docs)
    idf = (np.log((N + 1) / (df_counts + 1)) + 1.0)  # smoothed
    scores = tf * idf  # shape (n_docs, n_terms)

    return scores, terms

def _pick_top_phrases(scores: np.ndarray, terms: np.ndarray, k: int = 3) -> list[str]:
    out = []
    if scores.size == 0 or terms.size == 0:
        return out
    for row in scores:
        idx = np.argsort(row)[::-1]
        chosen = []
        for i in idx:
            t = terms[i]
            if len(t) <= 2:
                continue
            if t.isnumeric():
                continue
            # avoid trivial overlaps like "ndis" vs "ndis wa" dominating both
            if any(t in c or c in t for c in chosen):
                continue
            chosen.append(t)
            if len(chosen) >= k:
                break
        out.append(", ".join(chosen) if chosen else "misc")
    return out

def label_clusters(
    df: pd.DataFrame,
    text_col: str = "Query_norm",
    cluster_col: str = "cluster_id",
    weight_col: str | None = None,
    user_stopwords: list[str] | None = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Produce human-readable labels per cluster using c‑TF‑IDF with:
      - Expanded stopword list (to suppress generic words like 'near', 'service', etc.)
      - Optional importance weighting (e.g., by impressions)

    Returns a DataFrame with columns: [cluster_col, 'cluster_label']
    """
    if df.empty or text_col not in df.columns or cluster_col not in df.columns:
        return pd.DataFrame(columns=[cluster_col, "cluster_label"])

    cluster_ids = []
    docs = []
    weights = []

    # Build one "document" per cluster by concatenating normalized queries
    for cid, sub in df.groupby(cluster_col):
        cluster_ids.append(cid)
        docs.append(" ".join(sub[text_col].astype(str).tolist()))
        if weight_col and (weight_col in sub.columns):
            weights.append(float(sub[weight_col].fillna(0).sum()))
        else:
            weights.append(1.0)

    # IMPORTANT: pass list, not set
    stop_words_list = list(_build_stopwords(user_stopwords))

    scores, terms = _c_tfidf(docs, np.array(weights, dtype=float), stop_words_list)
    top_phrases = _pick_top_phrases(scores, terms, k=top_k)

    out = pd.DataFrame({
        cluster_col: cluster_ids,
        "cluster_label": top_phrases
    })

    # Title-case labels but keep acronyms
    out["cluster_label"] = out["cluster_label"].apply(lambda s: s.title() if isinstance(s, str) else s)
    return out
