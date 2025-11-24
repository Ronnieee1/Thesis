
from __future__ import annotations
import argparse
import os
import sys
import pandas as pd
import re
from typing import List, Tuple, Dict, Optional

# Third-party libs will be in requirements.txt: pandas, tldextract, validators
try:
    import tldextract
    import validators
except Exception:
    tldextract = None
    validators = None


# 1) Set standard column structures (lexical feats) dito / define our standardized lexical features
STANDARD_COLS: List[str] = [
    "id",
    "url",
    "url_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_underscores",
    "num_percent",
    "num_ampersands",
    "subdomain_count",
    "has_https",
    "http_in_hostname",
    "has_ip",
    "num_special",
    "contains_suspicious",
    "domain_entropy",
    "digit_count",
    "hyphen_count",
    "has_at_symbol",
    "double_slash_in_path",
    "has_dash_prefix_suffix",
    "has_multi_subdomains",
    "has_external_favicon",
    "scheme",
    "subdomain",
    "domain",
    "suffix",
    "netloc",
    "path",
    "query",
    "Class_Label",
]


# Utility: basic URL validator if validators not installed
_url_regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:\S+(?::\S*)?@)?"  # user:pass@
    r"(?:[A-Za-z0-9\-._~%]+|\[[A-Fa-f0-9:]+\])"  # domain or IPv6
    r"(?::\d{2,5})?"  # optional port
    r"(?:[/?#]\S*)?$",
    re.IGNORECASE,
)


def looks_like_url(s: str) -> bool:
    if s is None or not isinstance(s, str):
        return False
    s = s.strip()
    if s == "":
        return False
    if validators:
        try:
            return validators.url(s)
        except Exception:
            pass
    # fallback
    return bool(_url_regex.match(s))


# Semantic column detector: decide whether a column is list of URLs or lexical feats
# Returns: Tuple(is_url_col: bool, match_score_with_standard_cols: float)
def analyze_column(col: pd.Series, sample_size: int = 200) -> Tuple[bool, float]:
    """Analyze a single column.
    is_url_col: True if many values look like URLs
    match_score: fraction of STANDARD_COLS names that appear in dataset column names nearby -
                 we will compute this at DataFrame level (but function returns 0 for single col)
    """
    non_null = col.dropna()
    if len(non_null) == 0:
        return False, 0.0
    sample = non_null.astype(str).sample(min(len(non_null), sample_size), random_state=1)
    url_count = sum(1 for v in sample if looks_like_url(v))
    frac = url_count / len(sample)
    # heuristic: if > 0.6 look like urls -> URL column
    is_url = frac >= 0.6
    return is_url, 0.0


# DataFrame-level semantic checking
def detect_dataset_type(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (url_columns, matching_feature_columns)
    url_columns: columns detected as URL lists
    matching_feature_columns: columns whose names match STANDARD_COLS (case-insensitive)
    """
    url_cols = []
    for c in df.columns:
        is_url, _ = analyze_column(df[c])
        if is_url:
            url_cols.append(c)
    # Detect columns that match our standard features by name (exact or fuzzy)
    lname_to_col = {col.lower(): col for col in df.columns}
    matching = []
    for std in STANDARD_COLS:
        if std.lower() in lname_to_col:
            matching.append(lname_to_col[std.lower()])
    return url_cols, sorted(set(matching))


# Lexical feature extraction from single URL
import math
from collections import Counter


def compute_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [c / len(s) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def coerce_bool_to_int(series: pd.Series) -> pd.Series:
    """Convert a Series of mixed boolean-like values to 0/1 integers safely.
    Handles numeric, boolean, and common string representations.
    """
    # Try numeric conversion first
    num = pd.to_numeric(series, errors="coerce")
    result = pd.Series(0, index=series.index)
    mask_num = num.notna()
    if mask_num.any():
        result.loc[mask_num] = (num.loc[mask_num] != 0).astype(int)
    # For non-numeric, map common true/false strings
    non_num_idx = ~mask_num
    if non_num_idx.any():
        s = series.fillna("").astype(str).str.lower().str.strip()
        mapping = {"true": 1, "false": 0, "yes": 1, "no": 0, "y": 1, "n": 0, "t": 1, "f": 0, "1": 1, "0": 0}
        mapped = s.map(mapping).fillna(0).astype(int)
        result.loc[non_num_idx] = mapped.loc[non_num_idx]
    return result


def extract_features_from_url(url: str, source_id: Optional[object] = None) -> Dict[str, Optional[object]]:
    """Extract lexical features from a single URL string."""
    if url is None:
        url = ""
    url = str(url).strip()
    feat = {k: None for k in STANDARD_COLS}
    feat["id"] = source_id
    feat["url"] = url
    if url == "":
        for k in STANDARD_COLS:
            if feat.get(k) is None:
                # defaults: booleans False for has_/contains_/double_, ints for counts/lengths, empty string for textual
                if k.startswith("has_") or k.startswith("contains_") or k.startswith("double_"):
                    feat[k] = False
                elif any(x in k for x in ("count","num_","_length","_entropy","digit","hyphen")):
                    feat[k] = 0
                else:
                    feat[k] = ""
        return feat

    # parse using tldextract if available
    try:
        if tldextract:
            ext = tldextract.extract(url)
            subdomain = ext.subdomain or ""
            domain = ext.domain or ""
            suffix = ext.suffix or ""
        else:
            subdomain = ""
            domain = ""
            suffix = ""
    except Exception:
        subdomain = domain = suffix = ""

    # parse url parts
    from urllib.parse import urlparse

    p = urlparse(url)
    scheme = p.scheme or ""
    netloc = p.netloc or ""
    path = p.path or ""
    query = p.query or ""

    feat["scheme"] = scheme
    feat["netloc"] = netloc
    feat["path"] = path
    feat["query"] = query
    feat["subdomain"] = subdomain
    feat["domain"] = domain
    feat["suffix"] = suffix

    # requested lexical features
    feat["url_length"] = len(url)
    feat["path_length"] = len(path)
    feat["query_length"] = len(query)
    feat["num_dots"] = url.count('.')
    feat["num_underscores"] = url.count('_')
    feat["num_percent"] = url.count('%')
    feat["num_ampersands"] = url.count('&')
    feat["subdomain_count"] = 0 if subdomain == "" else len([s for s in subdomain.split('.') if s != ""])
    feat["has_https"] = scheme.lower() == "https"
    feat["http_in_hostname"] = 'http' in netloc.lower()
    feat["has_ip"] = bool(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", netloc))
    feat["num_special"] = sum(1 for c in url if not c.isalnum())
    suspicious_keywords = ["login","secure","account","verify","update","bank","confirm","password","wp-login","signin","payment","auth","credentials"]
    feat["contains_suspicious"] = any(k in url.lower() for k in suspicious_keywords)
    domain_for_entropy = (domain + ("." + suffix if suffix else "")).strip()
    feat["domain_entropy"] = compute_entropy(domain_for_entropy)
    feat["digit_count"] = sum(c.isdigit() for c in url)
    feat["hyphen_count"] = url.count('-')
    feat["has_at_symbol"] = '@' in url
    feat["double_slash_in_path"] = '//' in path
    feat["has_dash_prefix_suffix"] = '-' in domain
    feat["has_multi_subdomains"] = feat["subdomain_count"] > 1
    # heuristic for external favicon: favicon in url or path endswith .ico or contains 'icon'
    feat["has_external_favicon"] = ('favicon' in url.lower()) or path.lower().endswith('.ico') or ('icon' in path.lower())

    # default label: unknown/legitimate -> 0 (unless dataset provides labels later)
    feat["Class_Label"] = 0

    # coerce numeric types
    int_fields = ["url_length","path_length","query_length","num_dots","num_underscores","num_percent","num_ampersands","subdomain_count","num_special","digit_count","hyphen_count"]
    for f in int_fields:
        try:
            feat[f] = int(feat.get(f, 0))
        except Exception:
            feat[f] = 0
    try:
        feat["domain_entropy"] = float(feat.get("domain_entropy", 0.0))
    except Exception:
        feat["domain_entropy"] = 0.0

    return feat


# Preprocessing pipeline for URL columns
def preprocess_urls(df: pd.DataFrame, url_columns: List[str]) -> pd.DataFrame:
    """Given the df and list of url columns, extract lexical features from each URL and return
    a standardized DataFrame with STANDARD_COLS. If multiple URL columns exist, concatenate results.
    Also perform basic cleaning (drop duplicates, drop empty urls).
    """
    rows = []
    # For each input row, take the first URL-like value among the url_columns (if any)
    for idx, row in df.iterrows():
        found = None
        for col in url_columns:
            v = row.get(col)
            if pd.isna(v):
                continue
            sv = str(v).strip()
            if sv == "":
                continue
            if looks_like_url(sv):
                found = sv
                break
        if found:
            feat = extract_features_from_url(found, source_id=idx)
        else:
            # produce a default row with id and blank url
            feat = {k: None for k in STANDARD_COLS}
            feat["id"] = idx
            feat["url"] = ""
            # sensible defaults
            for c in STANDARD_COLS:
                if feat.get(c) is None:
                    if c.startswith("has_") or c.startswith("contains_") or c.startswith("double_"):
                        feat[c] = False
                    elif any(x in c for x in ("count","num_","_length","digit","hyphen","entropy")):
                        feat[c] = 0
                    else:
                        feat[c] = ""
        rows.append(feat)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=STANDARD_COLS)
    # ensure all standard cols present
    for c in STANDARD_COLS:
        if c not in out.columns:
            # default: False for has_/contains_/double_, 0 for counts/lengths, empty string otherwise
            if c.startswith("has_") or c.startswith("contains_") or c.startswith("double_"):
                out[c] = False
            elif any(x in c for x in ("count","num_","_length","digit","hyphen","entropy")):
                out[c] = 0
            else:
                out[c] = ""

    # coerce numeric types
    int_cols = [c for c in STANDARD_COLS if any(x in c for x in ("count","num_","_length","digit","hyphen"))]
    for c in int_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    if "domain_entropy" in out.columns:
        out["domain_entropy"] = pd.to_numeric(out["domain_entropy"], errors="coerce").fillna(0.0).astype(float)
    # booleans -> convert to 0/1
    bool_cols = [c for c in STANDARD_COLS if c.startswith("has_") or c.startswith("contains_") or c.startswith("double_")]
    for c in bool_cols:
        out[c] = coerce_bool_to_int(out[c])

    # Ensure Class_Label exists and is numeric 0/1 (default 0)
    if "Class_Label" not in out.columns:
        out["Class_Label"] = 0
    try:
        out["Class_Label"] = pd.to_numeric(out["Class_Label"], errors="coerce").fillna(0).astype(int)
    except Exception:
        out["Class_Label"] = coerce_bool_to_int(out["Class_Label"]).astype(int)

    # do not drop rows; preserve ids. ensure index reset
    out = out.reset_index(drop=True)
    return out[STANDARD_COLS]


# Preprocessing pipeline for already-extracted lexical features
def preprocess_lexical_features(df: pd.DataFrame, matching_cols: List[str], merge_strategy: str = "first", verbose: bool = False) -> pd.DataFrame:
    """Standardize column names, ensure all STANDARD_COLS exist, coerce types, clean rows.
    matching_cols: list of existing columns that map to STANDARD_COLS by name
    """
    # create a working copy
    w = df.copy()
    # lower-case column mapping
    col_map = {c: c for c in w.columns}
    # Build candidate source columns for each standard column using exact, substring, and fuzzy matching
    import difflib
    cols = list(w.columns)
    SIMILARITY_THRESHOLD = 0.6
    std_to_candidates: Dict[str, List[str]] = {}
    for std in STANDARD_COLS:
        std_l = std.lower()
        candidates: List[str] = []
        # exact
        candidates += [c for c in cols if c.lower() == std_l]
        # substring: be careful with short standard tokens (e.g., 'url') so we don't
        # match 'UrlLength' -> 'url'. Use word-boundary match for short std names.
        if len(std_l) < 4:
            # require word boundary match for short tokens
            candidates += [c for c in cols if re.search(rf"\b{re.escape(std_l)}\b", c.lower()) and c not in candidates]
        else:
            candidates += [c for c in cols if std_l in c.lower() and c not in candidates]
        # fuzzy: add those above threshold
        fuzzy_scores = [(c, difflib.SequenceMatcher(None, std_l, c.lower()).ratio()) for c in cols if c not in candidates]
        fuzzy_scores = sorted(fuzzy_scores, key=lambda x: x[1], reverse=True)
        for c, score in fuzzy_scores:
            if score >= SIMILARITY_THRESHOLD:
                candidates.append(c)
        std_to_candidates[std] = candidates

    # store mapping for debug
    mapping_used: Dict[str, List[str]] = {std: list(cands) for std, cands in std_to_candidates.items()}

    # Create/derive standard columns based on candidates and merge_strategy
    for std, candidates in std_to_candidates.items():
        if not candidates:
            continue
        if len(candidates) == 1:
            w[std] = w[candidates[0]].copy()
            continue
        # multiple candidates: merge according to strategy
        if merge_strategy == "first":
            w[std] = w[candidates[0]].copy()
        elif merge_strategy == "coalesce":
            # first non-null across candidates
            try:
                w[std] = w[candidates].bfill(axis=1).iloc[:, 0].copy()
            except Exception:
                w[std] = w[candidates[0]].copy()
        elif merge_strategy in ("sum", "mean"):
            num_df = w[candidates].apply(pd.to_numeric, errors="coerce").fillna(0)
            if merge_strategy == "sum":
                w[std] = num_df.sum(axis=1)
            else:
                w[std] = num_df.mean(axis=1)
        else:
            w[std] = w[candidates[0]].copy()

    # Ensure all standard cols exist and try to infer where possible
    for c in STANDARD_COLS:
        if c not in w.columns:
            candidates = [col for col in w.columns if c.lower() in col.lower()]
            if candidates:
                w[c] = w[candidates[0]]
            else:
                # sensible defaults
                if c.startswith("has_") or c.startswith("contains_") or c.startswith("double_"):
                    w[c] = False
                elif any(x in c for x in ("count","num_","_length","digit","hyphen","entropy")):
                    w[c] = 0
                else:
                    w[c] = ""

    # Coerce types for integer-like fields
    int_fields = [c for c in STANDARD_COLS if any(x in c for x in ("count","num_","_length","digit","hyphen"))]
    for n in int_fields:
        try:
            w[n] = pd.to_numeric(w[n], errors="coerce").fillna(0).astype(int)
        except Exception:
            w[n] = 0
    # float entropy
    if "domain_entropy" in w.columns:
        try:
            w["domain_entropy"] = pd.to_numeric(w["domain_entropy"], errors="coerce").fillna(0.0).astype(float)
        except Exception:
            w["domain_entropy"] = 0.0
    # booleans -> convert to 0/1
    bool_fields = [c for c in STANDARD_COLS if c.startswith("has_") or c.startswith("contains_") or c.startswith("double_")]
    for b in bool_fields:
        try:
            w[b] = coerce_bool_to_int(w[b])
        except Exception:
            w[b] = coerce_bool_to_int(w[b].astype(str))

    # Ensure url column exists and normalize (keep rows even if url is blank)
    if "url" in w.columns:
        w["url"] = w["url"].astype(str).fillna("").str.strip()
    else:
        w["url"] = ""

    # Ensure id exists (use original index if not provided)
    if "id" not in w.columns:
        w["id"] = w.index

    # Try to find label-like columns and map them to Class_Label (phishing=1, legitimate=0)
    label_candidates = [c for c in cols if re.search(r"label|class|phish|malicious|is_phish|is_malicious|y$|target", c, re.IGNORECASE)]
    if label_candidates:
        # pick the first reasonable candidate
        cand = label_candidates[0]
        try:
            w["Class_Label"] = pd.to_numeric(w[cand], errors="coerce").fillna(0).astype(int)
        except Exception:
            w["Class_Label"] = coerce_bool_to_int(w[cand]).astype(int)
    else:
        # default to 0
        w["Class_Label"] = 0

    # do not drop rows; preserve original row alignment
    if verbose:
        print("Column mapping (standard -> candidates):")
        for std, cands in mapping_used.items():
            if cands:
                sample_vals = []
                for c in cands[:3]:
                    try:
                        sample_vals.append(f"{c}[:3]={w[c].astype(str).fillna('')[:3].tolist()}")
                    except Exception:
                        sample_vals.append(f"{c} (no preview)")
                print(f"  {std}: {cands} -> samples: {sample_vals}")
    return w[STANDARD_COLS]


def finalize_and_write(df: pd.DataFrame, output: str) -> None:
    """Final defensive cleaning before writing output CSV.
    - Ensures all STANDARD_COLS exist
    - Coerces boolean-like values (True/False/yes/no/1/0 strings) to 0/1
    - Coerces numeric fields to numeric types
    - Ensures textual fields are strings and have no None
    - Writes CSV with explicit column order
    """
    out = df.copy()
    # Ensure all standard cols exist
    for c in STANDARD_COLS:
        if c not in out.columns:
            if c.startswith("has_") or c.startswith("contains_") or c.startswith("double_"):
                out[c] = 0
            elif any(x in c for x in ("count","num_","_length","digit","hyphen","entropy")):
                out[c] = 0
            else:
                out[c] = ""

    # Normalize numeric fields
    numeric_fields = [
        "url_length","path_length","query_length","num_dots","num_underscores",
        "num_percent","num_ampersands","subdomain_count","num_special","domain_entropy",
        "digit_count","hyphen_count",
    ]
    for c in numeric_fields:
        if c in out.columns:
            if c == "domain_entropy":
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # Ensure Class_Label is numeric 0/1 (last column)
    if "Class_Label" in out.columns:
        try:
            out["Class_Label"] = pd.to_numeric(out["Class_Label"], errors="coerce").fillna(0).astype(int)
        except Exception:
            out["Class_Label"] = coerce_bool_to_int(out["Class_Label"]).astype(int)
    else:
        out["Class_Label"] = 0

    # Normalize boolean-like fields to 0/1
    bool_cols = [col for col in STANDARD_COLS if col.startswith("has_") or col.startswith("contains_") or col.startswith("double_")]
    for c in bool_cols:
        if c in out.columns:
            out[c] = coerce_bool_to_int(out[c])

    # Ensure textual fields are strings and no None remains
    text_cols = [c for c in STANDARD_COLS if c not in numeric_fields and c not in bool_cols]
    for c in text_cols:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str)

    # Ensure sequential output ids: 1..N (user requested)
    # Coerce any existing id values to numeric then overwrite with sequential ids
    try:
        out["id"] = pd.to_numeric(out.get("id", pd.Series(range(len(out)))), errors="coerce")
    except Exception:
        out["id"] = pd.Series(range(len(out)))
    out["id"] = pd.Series(range(1, len(out) + 1), index=out.index).astype(int)

    # Reindex to standard order and write
    out = out.reindex(columns=STANDARD_COLS)
    out.to_csv(output, index=False)


# Main orchestrator

def process_dataset(path: str, output: Optional[str] = None, merge_strategy: str = "first", verbose: bool = False) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # load dataset
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        df = pd.read_csv(path)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(path)
    else:
        # try csv
        df = pd.read_csv(path)

    url_cols, matching = detect_dataset_type(df)
    print(f"Detected URL columns: {url_cols}")
    print(f"Detected matching lexical columns by name: {matching}")

    # Decision logic described by user:
    # - If detected URL column(s) AND there is NO overlap (by semantics) with our standard columns, call URL preprocessing.
    # - If detected columns are already our lexical feature columns, call lexical preprocessing.
    result_df = pd.DataFrame(columns=STANDARD_COLS)
    # Prefer URL-based preprocessing when a URL-like column exists.
    # This avoids treating a raw `url` column as already-preprocessed lexical features.
    if len(url_cols) > 0 and (len(matching) == 0 or any(col in url_cols for col in matching)):
        print("Running URL-based preprocessing pipeline...")
        result_df = preprocess_urls(df, url_cols)
    else:
        # if we detect any matching standard-like columns, treat dataset as preprocessed
        if len(matching) > 0:
            print("Detected some standard lexical columns — running lexical-features preprocessing pipeline...")
            result_df = preprocess_lexical_features(df, matching, merge_strategy=merge_strategy, verbose=verbose)
        elif len(url_cols) > 0:
            # ambiguous: there are URL columns but no clear matching lexical columns
            print("Ambiguous dataset: URL-like columns detected but no matching lexical columns — processing URLs.")
            result_df = preprocess_urls(df, url_cols)
        else:
            # no urls, no matching => nothing to do
            print("No URL columns detected and no matching lexical feature columns found. Returning empty standardized frame.")
            result_df = pd.DataFrame(columns=STANDARD_COLS)

    # output file
    if output is None:
        base, _ = os.path.splitext(path)
        output = base + "__lexical_standardized.csv"
    # Final defensive cleaning & write
    finalize_and_write(result_df, output)
    print(f"Wrote standardized lexical features to: {output}")
    return output


def process_folder(folder_path: str, output_dir: Optional[str] = None, merge_strategy: str = "first", verbose: bool = False) -> List[str]:
    """Process all CSV/XLS/XLSX/TXT files in folder_path.
    Writes one output file per input file into output_dir (created if missing).
    Returns list of output file paths.
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(folder_path)
    exts = {".csv", ".txt", ".xls", ".xlsx"}
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in exts
    ]
    if not files:
        print(f"No CSV/Excel/TXT files found in {folder_path}")
        return []

    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "Output URLs")
    os.makedirs(output_dir, exist_ok=True)

    outputs: List[str] = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(output_dir, base + "__lexical_standardized.csv")
        try:
            process_dataset(f, out_path, merge_strategy=merge_strategy, verbose=verbose)
            outputs.append(out_path)
        except Exception as e:
            print(f"Error processing {f}: {e}", file=sys.stderr)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dataset lexical feature standardizer (Taglish comments inside)")
    # make input optional; default to a folder named 'Inputed URLs' next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "Inputed URLs")
    parser.add_argument("input", nargs="?", default=default_input, help="Path to dataset file or a folder containing datasets. Defaults to the 'Inputed URLs' folder next to the script.")
    parser.add_argument("-o", "--output", help="Output CSV path (optional). If input is a folder and this is a folder path, outputs are written there. If omitted, defaults to 'Output URLs' next to the script.")
    parser.add_argument("--merge-strategy", choices=["first", "coalesce", "sum", "mean"], default="first", help="How to merge multiple candidate columns when mapping preprocessed datasets (default: first)")
    parser.add_argument("--verbose", action="store_true", help="Print mapping and small samples when standardizing preprocessed datasets")
    args = parser.parse_args(argv)

    inp = args.input
    # if input is a directory, process all files inside and write outputs to output folder
    if os.path.isdir(inp):
        out_dir = args.output if args.output is not None else os.path.join(script_dir, "Output URLs")
        os.makedirs(out_dir, exist_ok=True)
        outputs = process_folder(inp, out_dir, merge_strategy=args.merge_strategy, verbose=args.verbose)
        if outputs:
            print(f"Processed {len(outputs)} files. Outputs written to: {out_dir}")
        else:
            print("No files processed.")
    else:
        # single file
        # determine output path: if -o given and is a directory, write inside it; if -o given as a file write there; else write into Output URLs folder
        if args.output:
            # if user provided an output directory
            if os.path.isdir(args.output) or args.output.endswith(os.sep):
                os.makedirs(args.output, exist_ok=True)
                base = os.path.splitext(os.path.basename(inp))[0]
                out_path = os.path.join(args.output, base + "__lexical_standardized.csv")
            else:
                out_path = args.output
        else:
            out_dir = os.path.join(script_dir, "Output URLs")
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(inp))[0]
            out_path = os.path.join(out_dir, base + "__lexical_standardized.csv")

        out = process_dataset(inp, out_path, merge_strategy=args.merge_strategy, verbose=args.verbose)
        print("Done.")


if __name__ == "__main__":
    main()
