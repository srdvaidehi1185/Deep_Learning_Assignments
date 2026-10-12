import json
from pathlib import Path

import pandas as pd

from graph_builder import build_graph

DATA_DIR = Path("data/processed")
SPLITS = ["train", "validation", "test"]


for split in SPLITS:
    csv_path = DATA_DIR / f"{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run prepare_data.py first."
        )

    df = pd.read_csv(csv_path)
    output_path = DATA_DIR / f"{split}_graphs.jsonl"

    print(f"Building graphs for {split}: {len(df)} rows")

    with output_path.open("w", encoding="utf-8") as output:
        for i, row in df.iterrows():
            graph = build_graph(str(row["code"]))
            graph["label"] = int(row["label"])
            output.write(json.dumps(graph) + "\n")

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(df)}")

    print(f"Saved graphs: {output_path}")

print("Graph construction completed.")
