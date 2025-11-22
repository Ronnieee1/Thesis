Dataset Lexical Feature Standardizer (Tagalog + English)

What this project does

- Loads a dataset (CSV or Excel).
- Detects whether the dataset contains columns that are lists of URLs or already-extracted lexical features.
- If URL columns are found (and no existing lexical feature columns detected), it extracts lexical features from each URL (scheme, domain, path features, length, entropy, etc.).
- If lexical feature columns are already present, it standardizes their names/types to a predefined standard.
- Outputs a standardized CSV with a fixed set of lexical feature columns.

Standard columns

The default standardized columns are:

- url, scheme, subdomain, domain, suffix, path, query, params, netloc, length, num_subdirs, has_ip, num_digits, num_letters, num_special, entropy

Usage

1) Create and activate a Python environment (recommended):

# in PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2) Install requirements

pip install -r requirements.txt

3) Run the program

python main.py path\to\your_dataset.csv

Optional:

python main.py path\to\your_dataset.csv -o output_standardized.csv

Notes and assumptions (Tagalog + English)

- Ito ay isang maliit na helper script para i-standardize ang lexical features mula sa raw URLs o mula sa dati nang-extract na features.
- Ginagamit nito ang heuristic na 60% ng sampled non-null values ng isang column ay URL-like para ma-classify as URL column.
- Sa ambiguity (parehas may URL columns at may ilang matching feature columns), ire-process muna ang URL column at i-merge sa umiiral na features kung possible.
- Kung may kulang na standard columns, pipunan ng default values (empty string or 0).
- Hindi ni-validate nang husto ang lahat ng edge cases. Feel free to open an issue or request enhancements.

Next steps / Improvements

- Add unit tests for the two pipelines.
- Add more robust semantic matching (fuzzy name matching, type checks).
- Add concurrency for speed when processing large URL lists.
- Integrate CLI flags to customize standard columns.


Author: Automated assistant — created for your Thesis workspace
