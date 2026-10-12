from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": "train-00000-of-00001-396a063c42dfdb0a.parquet",
    "validation": "validation-00000-of-00001-5d4ba937305086b9.parquet",
    "test": "test-00000-of-00001-e0e162fa10729371.parquet",
}

CODE_COLUMNS = [
    "func", "function", "code", "source", "content",
    "function_code", "src"
]
LABEL_COLUMNS = [
    "target", "label", "labels", "is_vulnerable",
    "vulnerable", "y"
]


def find_column(columns, candidates):
    lookup = {str(c).lower(): c for c in columns}

    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def convert_label(value):
    """Convert common binary vulnerability labels into 0 or 1."""
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        if value in (0, 1):
            return int(value)
        return None

    text = str(value).strip().lower()

    vulnerable_values = {
        "1", "true", "yes", "vulnerable", "positive", "buggy"
    }
    safe_values = {
        "0", "false", "no", "non-vulnerable",
        "non_vulnerable", "safe", "negative", "clean"
    }

    if text in vulnerable_values:
        return 1

    if text in safe_values:
        return 0

    return None


for split, filename in SPLITS.items():
    source = RAW_DIR / filename

    if not source.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {source}\n"
            "Copy the downloaded Parquet file into data/raw/."
        )

    print(f"\nReading {split}: {source}")
    df = pd.read_parquet(source)

    print("Rows and columns:", df.shape)
    print("Available columns:", df.columns.tolist())

    code_col = find_column(df.columns, CODE_COLUMNS)
    label_col = find_column(df.columns, LABEL_COLUMNS)

    if code_col is None or label_col is None:
        raise ValueError(
            f"Could not detect the code/label columns in {filename}.\n"
            f"Columns found: {df.columns.tolist()}\n"
            "Edit CODE_COLUMNS or LABEL_COLUMNS to match your dataset."
        )

    result = pd.DataFrame({
        "code": df[code_col].fillna("").astype(str),
        "label": df[label_col].apply(convert_label),
    })

    result = result[
        (result["code"].str.strip() != "") &
        (result["label"].isin([0, 1]))
    ].copy()

    result["label"] = result["label"].astype(int)
    result.reset_index(drop=True, inplace=True)

    if result.empty:
        raise ValueError(
            f"No valid rows remain in {split}. "
            "Check the detected label values and mapping."
        )

    output = OUT_DIR / f"{split}.csv"
    result.to_csv(output, index=False)

    print(f"Saved: {output}")
    print("Rows kept:", len(result))
    print("Label counts:")
    print(result["label"].value_counts().sort_index().to_dict())

print("\nDataset preparation completed.")
print("Label convention used: 0 = non-vulnerable, 1 = vulnerable.")
print("Verify this convention against your dataset documentation.")
