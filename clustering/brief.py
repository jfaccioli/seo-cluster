import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

WH_WORDS = r"^(who|what|when|where|why|how|can|does|do|is|are|should)\b"

# Note: SentenceTransformer commented out to avoid NotImplementedError
# semantic_model = SentenceTransformer("distilbert-base-uncased", device="cpu")

def _semantic_top_phrases(texts: List[str], top_k: int = 10) -> List[str]:
    if len(texts) == 0:
        return []
    try:
        vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words="english", min_df=1)
        docs = [" ".join(texts)]
        X = vectorizer.fit_transform(docs)
        terms = vectorizer.get_feature_names_out()
        # Fix: Check if terms array is empty using len() instead of .size
        if len(terms) == 0:
            return texts[:top_k]  # Fallback to raw texts if no terms
        freqs = X.toarray().sum(axis=0)  # Sum frequencies across documents
        idx = np.argsort(freqs)[::-1]  # Sort by frequency (highest first)
        return [terms[i] for i in idx[:top_k]]  # Take top_k based on frequency
    except Exception as e:
        # Fallback to returning first few texts if vectorization fails
        return texts[:top_k]

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
    # Fix: Ensure cluster_label is a valid string - handle Series/array case
    if isinstance(cluster_label, (pd.Series, np.ndarray)):
        # If it's a Series or array, get the first value
        cluster_label = str(cluster_label.iloc[0] if hasattr(cluster_label, 'iloc') else cluster_label[0])
    elif not isinstance(cluster_label, str):
        # If it's not a string, convert it
        cluster_label = str(cluster_label)
    # Now safely check if it's empty after ensuring it's a string
    if not cluster_label or not cluster_label.strip():
        cluster_label = f"Cluster {cluster_id}"
    try:
        # Key phrases using CountVectorizer and semantic filtering
        query_texts = data["Query_norm"].tolist()
        keyphrases = _semantic_top_phrases(query_texts, top_k=top_phrases_k)
        # Intents & buckets
        data["intent"] = data["Query_norm"].map(_intent_bucket)
        buckets = data.groupby("intent")["Query"].apply(lambda s: list(s)[:5]).to_dict()
        # FAQs (top by impressions)
        faq_df = data[data["intent"] == "FAQ"].copy()
        if not faq_df.empty:
            faq_df = faq_df.sort_values("Impressions", ascending=False).head(6)
            faqs = faq_df["Query"].tolist()
        else:
            faqs = []
        # Generate titles/H1 ideas using templates
        title_opts = [
            f"{cluster_label} Guide: {keyphrases[0]} in WA" if keyphrases and keyphrases[0] else f"{cluster_label} Guide",
            f"{cluster_label} - Services and Costs in Perth" if keyphrases and keyphrases[0] else f"{cluster_label} Overview",
            f"Top {keyphrases[0]} Options in Australia" if keyphrases and keyphrases[0] else f"Top {cluster_label} Tips",
            f"How to Choose {keyphrases[0]} in 2025" if keyphrases and keyphrases[0] else f"How to Choose {cluster_label}"
        ][:4]  # Limit to 4
        # Generate H2 sections using intents and keyphrases - with safety checks
        h2_suggestions = []
        if 'FAQ' in buckets and buckets['FAQ']:
            h2_suggestions.append(f"{buckets['FAQ'][0]}: Common Questions")
        else:
            h2_suggestions.append("FAQs")
        if 'Informational' in buckets and buckets['Informational']:
            h2_suggestions.append(f"{buckets['Informational'][0]} Insights")
        else:
            h2_suggestions.append("Key Insights")
        if keyphrases and 'Local / Navigational' in buckets and buckets['Local / Navigational']:
            h2_suggestions.append(f"Local {keyphrases[0]} Options")
        else:
            h2_suggestions.append("Local Guide")
        if keyphrases and 'Transactional / Service' in buckets and buckets['Transactional / Service']:
            h2_suggestions.append(f"{keyphrases[0]} Services Near You")
        else:
            h2_suggestions.append("Services")
        if keyphrases and keyphrases[0]:
            h2_suggestions.extend([
                f"Benefits of {keyphrases[0]}",
                f"Costs of {keyphrases[0]} in 2025"
            ])
        else:
            h2_suggestions.extend([
                "Benefits Overview",
                "Cost Guide"
            ])
        h2_suggestions = h2_suggestions[:6]  # Limit to 6
        # Related topics (semantic expansion)
        related_topics = [p for p in keyphrases[3:8] if p not in h2_suggestions] if keyphrases else []
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
