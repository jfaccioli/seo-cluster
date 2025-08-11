import re
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\-]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s
