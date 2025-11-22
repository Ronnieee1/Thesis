
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
    "url",
    "scheme",
    "subdomain",
    "domain",
    "suffix",
    "path",
    "query",
    "params",
    "netloc",
    "length",
    "num_subdirs",
    "has_ip",
    "num_digits",
    "num_letters",
    "num_special",
    "entropy",
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


def extract_features_from_url(url: str) -> Dict[str, Optional[object]]:
    """Extract lexical features from a single URL string."""
    if url is None:
        url = ""
    url = str(url).strip()
    feat = {k: None for k in STANDARD_COLS}
    feat["url"] = url
    if url == "":
        # fill defaults
        for k in STANDARD_COLS:
            if feat.get(k) is None:
                feat[k] = "" if isinstance(k, str) else 0
        return feat

    # parse using tldextract if available
    try:
        if tldextract:
            ext = tldextract.extract(url)
            feat["subdomain"] = ext.subdomain
            feat["domain"] = ext.domain
            feat["suffix"] = ext.suffix
        else:
            # fallback: very naive
            feat["subdomain"] = ""
            feat["domain"] = ""
            feat["suffix"] = ""
    except Exception:
        feat["subdomain"] = feat["domain"] = feat["suffix"] = ""

    # scheme and netloc/path parts using urlparse
    try:
        from urllib.parse import urlparse, parse_qs

        p = urlparse(url)
        feat["scheme"] = p.scheme
        feat["netloc"] = p.netloc
        feat["path"] = p.path
        feat["query"] = p.query
        feat["params"] = p.params
        # path parts
        path = p.path or ""
        if path.startswith("/"):
            path = path[1:]
        feat["num_subdirs"] = 0 if path == "" else len([p for p in path.split("/") if p != ""])
    except Exception:
        feat["scheme"] = feat["netloc"] = feat["path"] = feat["query"] = feat["params"] = ""
        feat["num_subdirs"] = 0

    # lengths and counts
    feat["length"] = len(url)
    feat["has_ip"] = bool(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", url))
    feat["num_digits"] = sum(c.isdigit() for c in url)
    feat["num_letters"] = sum(c.isalpha() for c in url)
    feat["num_special"] = sum(not c.isalnum() for c in url)
    feat["entropy"] = compute_entropy(url)

    # convert numeric fields to int/float
    try:
        feat["num_subdirs"] = int(feat.get("num_subdirs", 0))
    except Exception:
        feat["num_subdirs"] = 0
    try:
        feat["length"] = int(feat.get("length", 0))
    except Exception:
        feat["length"] = 0

    return feat


# Preprocessing pipeline for URL columns
def preprocess_urls(df: pd.DataFrame, url_columns: List[str]) -> pd.DataFrame:
    """Given the df and list of url columns, extract lexical features from each URL and return
    a standardized DataFrame with STANDARD_COLS. If multiple URL columns exist, concatenate results.
    Also perform basic cleaning (drop duplicates, drop empty urls).
    """
    rows = []
    for col in url_columns:
        for v in df[col].dropna().astype(str):
            if not looks_like_url(v):
                continue
            feat = extract_features_from_url(v)
            rows.append(feat)
    out = pd.DataFrame(rows)
    if out.empty:
        # return empty standardized frame
        return pd.DataFrame(columns=STANDARD_COLS)
    # ensure all standard cols present
    for c in STANDARD_COLS:
        if c not in out.columns:
            out[c] = "" if c not in ["length","num_subdirs","has_ip","num_digits","num_letters","num_special","entropy"] else 0
    # type fixes
    out["length"] = out["length"].astype(int)
    out["num_subdirs"] = out["num_subdirs"].astype(int)
    out["has_ip"] = out["has_ip"].astype(bool)
    out["num_digits"] = out["num_digits"].astype(int)
    out["num_letters"] = out["num_letters"].astype(int)
    out["num_special"] = out["num_special"].astype(int)
    out["entropy"] = out["entropy"].astype(float)

    # cleaning: drop duplicates by url
    out = out.drop_duplicates(subset=["url"])
    out = out.reset_index(drop=True)
    return out[STANDARD_COLS]


# Preprocessing pipeline for already-extracted lexical features
def preprocess_lexical_features(df: pd.DataFrame, matching_cols: List[str]) -> pd.DataFrame:
    """Standardize column names, ensure all STANDARD_COLS exist, coerce types, clean rows.
    matching_cols: list of existing columns that map to STANDARD_COLS by name
    """
    # create a working copy
    w = df.copy()
    # lower-case column mapping
    col_map = {c: c for c in w.columns}
    # Try to map existing columns to standard names (case-insensitive)
    # Use fuzzy matching (difflib) when exact/substring matches fail.
    import difflib
    cols = list(w.columns)
    rename_map = {}
    SIMILARITY_THRESHOLD = 0.6
    for std in STANDARD_COLS:
        std_l = std.lower()
        # 1) exact case-insensitive match
        exact = [c for c in cols if c.lower() == std_l]
        if exact:
            rename_map[exact[0]] = std
            continue
        # 2) substring match
        substr = [c for c in cols if std_l in c.lower()]
        if substr:
            rename_map[substr[0]] = std
            continue
        # 3) fuzzy match using difflib
        best = None
        best_score = 0.0
        for c in cols:
            score = difflib.SequenceMatcher(None, std_l, c.lower()).ratio()
            if score > best_score:
                best_score = score
                best = c
        if best and best_score >= SIMILARITY_THRESHOLD:
            rename_map[best] = std
    if rename_map:
        w = w.rename(columns=rename_map)

    # Ensure all standard cols exist
    for c in STANDARD_COLS:
        if c not in w.columns:
            # try to infer from partial names
            candidates = [col for col in w.columns if c.lower() in col.lower()]
            if candidates:
                w[c] = w[candidates[0]]
            else:
                w[c] = "" if c not in ["length","num_subdirs","has_ip","num_digits","num_letters","num_special","entropy"] else 0

    # Coerce types
    numeric_ints = ["length","num_subdirs","num_digits","num_letters","num_special"]
    for n in numeric_ints:
        try:
            w[n] = pd.to_numeric(w[n], errors="coerce").fillna(0).astype(int)
        except Exception:
            w[n] = 0
    for n in ["entropy"]:
        try:
            w[n] = pd.to_numeric(w[n], errors="coerce").fillna(0.0).astype(float)
        except Exception:
            w[n] = 0.0
    # has_ip to bool
    try:
        w["has_ip"] = w["has_ip"].astype(bool)
    except Exception:
        w["has_ip"] = w["has_ip"].apply(lambda x: bool(x) if pd.notna(x) else False)

    # Basic cleaning: drop rows with empty url and dedupe
    if "url" in w.columns:
        w["url"] = w["url"].astype(str)
        w = w[w["url"].str.strip() != ""]
        w = w.drop_duplicates(subset=["url"]).reset_index(drop=True)
    # ensure column order
    return w[STANDARD_COLS]


# Main orchestrator

def process_dataset(path: str, output: Optional[str] = None) -> str:
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
    if len(url_cols) > 0 and len(matching) == 0:
        print("Running URL-based preprocessing pipeline...")
        result_df = preprocess_urls(df, url_cols)
    else:
        # if matching covers a reasonable portion of STANDARD_COLS, do lexical preprocessing
        if len(matching) >= max(1, int(0.3 * len(STANDARD_COLS))):
            print("Running lexical-features preprocessing pipeline...")
            result_df = preprocess_lexical_features(df, matching)
        elif len(url_cols) > 0:
            # ambiguous: there are URL columns but also some standard-like columns
            print("Ambiguous dataset: both URL-like columns and some standard lexical columns. We'll process URLs and then merge/overwrite with lexical columns when present.")
            urls_df = preprocess_urls(df, url_cols)
            lex_df = preprocess_lexical_features(df, matching)
            # merge on url if present, else concatenate side-by-side
            if "url" in lex_df.columns and not lex_df.empty:
                merged = pd.concat([urls_df.set_index("url"), lex_df.set_index("url")], axis=1, join="outer")
                merged = merged.reset_index()
                # take lex_df values when present
                for c in STANDARD_COLS:
                    if c in lex_df.columns and c in merged.columns:
                        # prefer lex_df non-null
                        merged[c] = merged[c].combine_first(merged[c])
                result_df = merged[STANDARD_COLS]
            else:
                # just concatenate
                result_df = pd.concat([urls_df, lex_df], ignore_index=True, sort=False)[STANDARD_COLS]
        else:
            # no urls, no matching => nothing to do
            print("No URL columns detected and no matching lexical feature columns found. Returning empty standardized frame.")
            result_df = pd.DataFrame(columns=STANDARD_COLS)

    # Final cleaning: ensure types and order
    for c in ["length","num_subdirs","num_digits","num_letters","num_special"]:
        if c in result_df.columns:
            result_df[c] = pd.to_numeric(result_df[c], errors="coerce").fillna(0).astype(int)
    if "entropy" in result_df.columns:
        result_df["entropy"] = pd.to_numeric(result_df["entropy"], errors="coerce").fillna(0.0).astype(float)
    if "has_ip" in result_df.columns:
        result_df["has_ip"] = result_df["has_ip"].astype(bool)

    # output file
    if output is None:
        base, _ = os.path.splitext(path)
        output = base + "__lexical_standardized.csv"
    result_df.to_csv(output, index=False)
    print(f"Wrote standardized lexical features to: {output}")
    return output


def process_folder(folder_path: str, output_dir: Optional[str] = None) -> List[str]:
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
            process_dataset(f, out_path)
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
    args = parser.parse_args(argv)

    inp = args.input
    # if input is a directory, process all files inside and write outputs to output folder
    if os.path.isdir(inp):
        out_dir = args.output if args.output is not None else os.path.join(script_dir, "Output URLs")
        os.makedirs(out_dir, exist_ok=True)
        outputs = process_folder(inp, out_dir)
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

        out = process_dataset(inp, out_path)
        print("Done.")


if __name__ == "__main__":
    main()
