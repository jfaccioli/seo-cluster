from typing import List
import numpy as np
def _load_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None
_model = None
def embed_queries(queries: List[str]) -> np.ndarray:
    global _model
    if _model is None:
        _model = _load_model()
    if _model is not None:
        return np.array(_model.encode(queries, show_progress_bar=False, normalize_embeddings=True))
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X = vec.fit_transform(queries)
    return X.toarray()
