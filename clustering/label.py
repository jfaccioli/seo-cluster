import pandas as pd
from collections import Counter
def label_clusters(df: pd.DataFrame, text_col: str, cluster_col: str) -> pd.DataFrame:
    docs, ids = [], []
    for cid, sub in df.groupby(cluster_col):
        ids.append(cid)
        docs.append(" ".join(sub[text_col].astype(str)))
    if not docs:
        return pd.DataFrame(columns=[cluster_col, "cluster_label"])
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        import numpy as np
        cv = CountVectorizer(ngram_range=(1,3), stop_words="english", min_df=1)
        X = cv.fit_transform(docs)
        tf = X.toarray()
        df_counts = (X > 0).sum(axis=0).A1
        N = len(docs)
        c_tfidf = tf * (np.log((N + 1) / (df_counts + 1)) + 1)
        terms = np.array(cv.get_feature_names_out())
        labels = []
        for row in c_tfidf:
            idx = row.argsort()[-3:][::-1]
            top_terms = [t for t in terms[idx] if len(t) > 1][:3]
            label = ", ".join(top_terms) if top_terms else "misc"
            labels.append(label)
        return pd.DataFrame({cluster_col: ids, "cluster_label": labels})
    except Exception:
        labels = []
        for cid, sub in df.groupby(cluster_col):
            common = Counter(" ".join(sub[text_col]).split()).most_common(3)
            labels.append((cid, ", ".join([w for w,_ in common])))
        return pd.DataFrame(labels, columns=[cluster_col, "cluster_label"])
