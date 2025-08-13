import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

WH_WORDS = r"^(who|what|when|where|why|how|can|does|do|is|are|should)\b"

# Initialize SentenceTransformer for semantic embeddings
semantic_model = SentenceTransformer("distilbert-base-uncased")

def _semantic_top_phrases(texts: List[str], top_k: int = 10) -> List[str]:
    if len(texts) == 0:
        return []
    embeddings = semantic_model.encode(texts, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1)  # Use norm as a simple relevance metric
    idx = np.argsort(norms)[::-1]  # Sort by norm (highest first)
    tops = [texts[i] for i in idx[:top_k*2]]
    out = []
    for t in tops:
        if not any(str(t) in str(o) or str(o) in str(t) for o in out if isinstance(o, str)):
            out.append(t)
        if len(out) >= top_k:
            break
    return out

def _intent_bucket(q: str) -> str:
    s = q.lower()
    if re.search(WH_WORDS, s) or s.endswith("?"):
        return "FAQ"
    if any(x in s for x in ["near me", "perth", "wa", "closest", "local"]):
        return "Local / Navigational"
    if any(x in s for x in ["price", "cost", "quote", "book", "apply", "eligibility", "provider", "service"]):
        return "Transactional / Service"
    if any(x in s for x in ["compare", "vs", "best", "top", "reviews"]):
        return "Comparative"
    return "Informational"

def _suggest_page_type(intents: pd.Series, cluster_size: int) -> Tuple[str, int, int]:
    counts = intents.value_counts()
    faq_ratio = counts.get("FAQ", 0) / max(cluster_size, 1)
    trans_ratio = counts.get("Transactional / Service", 0) / max(cluster_size, 1)
    info_ratio = counts.get("Informational", 0) / max(cluster_size, 1)

    if trans_ratio >= 0.35:
        return ("Service / Landing Page", 700, 1200)
    if faq_ratio >= 0.35:
        return ("FAQ / Help Guide", 800, 1500)
    if cluster_size > 20 and info_ratio >= 0.5:
        return ("Pillar / In-depth Guide", 1200, 2000)
    return ("Article / Guide", 900, 1500)

def _format_md_list(items: List[str]) -> str:
    return "\n".join(f"- {x}" for x in items)

def nearest_clusters(
    centroids: Dict[int, np.ndarray],
    target_id: int,
    top_n: int = 5
) -> List[int]:
    if target_id not in centroids:
        return []
    keys = [k for k in centroids.keys() if k != target_id and k != -1]
    if not keys:
        return []
    A = centroids[target_id]
    sims = []
    for k in keys:
        B = centroids[k]
        if A is None or B is None:
            continue
        denom = (np.linalg.norm(A) * np.linalg.norm(B)) or 1.0
        sims.append((k, float(np.dot(A, B)/denom)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in sims[:top_n]]

def build_content_brief(
    df_cluster: pd.DataFrame,
    cluster_id: int,
    cluster_label: str,
    centroids: Dict[int, np.ndarray] | None = None,
    top_phrases_k: int = 10
) -> str:
    """Returns a Markdown brief using ML-based generation."""
    data = df_cluster.copy()
    if data.empty:
        return "# Content Brief\n\n_No data in this cluster._"

    # Ensure cluster_label is a valid string
    if not isinstance(cluster_label, str) or not cluster_label.strip():
        cluster_label = f"Cluster {cluster_id}"

    try:
        # Key phrases using CountVectorizer and semantic filtering
        vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words="english", min_df=1)
        docs = [" ".join(data["Query_norm"].tolist())]
        X = vectorizer.fit_transform(docs)
        terms = vectorizer.get_feature_names_out() if X.shape[1] > 0 else []
        keyphrases = _semantic_top_phrases(terms if terms else data["Query_norm"].tolist(), top_k=top_phrases_k)

        # Intents & buckets
        data["intent"] = data["Query_norm"].map(_intent_bucket)
        buckets = data.groupby("intent")["Query"].apply(lambda s: list(s)[:5]).to_dict()

        # FAQs (top by impressions)
        faq_df = data[data["intent"] == "FAQ"].copy()
        faq_df = faq_df.sort_values("Impressions", ascending=False).head(6)
        faqs = faq_df["Query"].tolist()

        # Generate titles/H1 ideas using templates
        title_opts = [
            f"{cluster_label} Guide: {keyphrases[0]} in WA" if keyphrases else f"{cluster_label} Guide",
            f"{cluster_label} - Services and Costs in Perth" if keyphrases else f"{cluster_label} Overview",
            f"Top {keyphrases[0]} Options in Australia" if keyphrases else f"Top {cluster_label} Tips",
            f"How to Choose {keyphrases[0]} in 2025" if keyphrases else f"How to Choose {cluster_label}"
        ][:4]  # Limit to 4

        # Generate H2 sections using intents and keyphrases
        h2_suggestions = [
            f"{buckets.get('FAQ', ['FAQ'])[0]}: Common Questions" if 'FAQ' in buckets else "FAQs",
            f"{buckets.get('Informational', ['Info'])[0]} Insights" if 'Informational' in buckets else "Key Insights",
            f"Local {keyphrases[0]} Options" if keyphrases and 'Local / Navigational' in buckets else "Local Guide",
            f"{keyphrases[0]} Services Near You" if keyphrases and 'Transactional / Service' in buckets else "Services",
            f"Benefits of {keyphrases[0]}" if keyphrases else "Benefits Overview",
            f"Costs of {keyphrases[0]} in 2025" if keyphrases else "Cost Guide"
        ][:6]  # Limit to 6

        # Related topics (semantic expansion)
        related_topics = [p for p in keyphrases[3:8] if p not in h2_suggestions]

        # Page type & word count
        page_type, min_words, max_words = _suggest_page_type(data["intent"], len(data))

        # Internal links
        link_ids = []
        if centroids:
            link_ids = nearest_clusters(centroids, cluster_id, top_n=5)

        # Build Markdown
        md = []
        md.append(f"# Content Brief: {cluster_label}")
        md.append("")
        md.append(f"**Cluster ID:** {cluster_id}")
        md.append(f"**Queries:** {len(data)}  ·  **Clicks:** {int(data['Clicks'].sum())}  ·  **Impressions:** {int(data['Impressions'].sum())}  ·  **Avg Pos:** {round(data['Position'].mean(),1) if not np.isnan(data['Position'].mean()) else '—'}")
        md.append(f"**Recommended Page Type:** {page_type}  ·  **Suggested Length:** {min_words}–{max_words} words")
        md.append("")
        md.append("## Title / H1 Ideas")
        md.append(_format_md_list(title_opts) if title_opts else "_(auto-generate after content)_")
        md.append("")
        md.append("## H2 / Sections to Cover")
        md.append(_format_md_list(h2_suggestions) if h2_suggestions else "_(derive from queries)_")
        md.append("")
        md.append("## Key Phrases to Work In")
        md.append(_format_md_list([p.title() for p in keyphrases]) if keyphrases else "_(auto)_")
        md.append("")
        if related_topics:
            md.append("## Related Topics to Consider")
            md.append(_format_md_list([p.title() for p in related_topics]))
            md.append("")
        if faqs:
            md.append("## FAQs to Answer")
            md.append(_format_md_list([q.rstrip("?") + "?" for q in faqs]))
            md.append("")
        if link_ids:
            md.append("## Internal Link Suggestions (related clusters)")
            md.append(_format_md_list([f"Cluster {cid}" for cid in link_ids]))
            md.append("")
        md.append("## Example Queries in this Cluster")
        md.append(_format_md_list(data["Query"].head(10).tolist()))
        return "\n".join(md)
    except Exception as e:
        return f"# Content Brief: {cluster_label}\n\n**Error**: Failed to generate brief due to {str(e)} for cluster with {len(data)} queries. Please ensure the cluster has valid data."
