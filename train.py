import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from model import GGNNClassifier

DATA_DIR = Path("data/processed")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
HIDDEN_DIM = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


class GraphDataset(Dataset):
    def __init__(self, filename):
        self.graphs = []

        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.graphs.append(json.loads(line))

        if not self.graphs:
            raise ValueError(f"No graphs found in {filename}")

        print(f"Loaded {len(self.graphs)} graphs from {filename}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def collate_graphs(graphs):
    all_x = []
    all_edges = []
    all_edge_types = []
    all_graph_ids = []
    labels = []

    offset = 0

    for graph_id, graph in enumerate(graphs):
        x = torch.tensor(graph["x"], dtype=torch.float32)
        all_x.append(x)

        edges = graph["edge_index"]
        edge_types = graph["edge_type"]

        if edges:
            edge_tensor = torch.tensor(
                edges, dtype=torch.long
            ).t().contiguous()
            edge_tensor += offset
            all_edges.append(edge_tensor)

            all_edge_types.append(
                torch.tensor(edge_types, dtype=torch.long)
            )

        all_graph_ids.append(
            torch.full((x.size(0),), graph_id, dtype=torch.long)
        )

        labels.append(int(graph["label"]))
        offset += x.size(0)

    x = torch.cat(all_x, dim=0)
    graph_ids = torch.cat(all_graph_ids, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    if all_edges:
        edge_index = torch.cat(all_edges, dim=1)
        edge_type = torch.cat(all_edge_types, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    return x, edge_index, edge_type, graph_ids, labels


def evaluate(model, loader):
    model.eval()
    actual = []
    predicted = []

    with torch.no_grad():
        for x, edges, edge_types, graph_ids, labels in loader:
            x = x.to(DEVICE)
            edges = edges.to(DEVICE)
            edge_types = edge_types.to(DEVICE)
            graph_ids = graph_ids.to(DEVICE)

            logits = model(x, edges, edge_types, graph_ids)
            predictions = logits.argmax(dim=1).cpu().tolist()

            actual.extend(labels.tolist())
            predicted.extend(predictions)

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).tolist(),
    }


train_data = GraphDataset(DATA_DIR / "train_graphs.jsonl")
val_data = GraphDataset(DATA_DIR / "validation_graphs.jsonl")
test_data = GraphDataset(DATA_DIR / "test_graphs.jsonl")

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE,
    shuffle=True, collate_fn=collate_graphs
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate_graphs
)
test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate_graphs
)

train_labels = [g["label"] for g in train_data.graphs]
class_counts = np.bincount(train_labels, minlength=2)

if (class_counts == 0).any():
    raise ValueError(
        "Training data must contain both labels 0 and 1. "
        f"Counts: {class_counts.tolist()}"
    )

class_weights = len(train_labels) / (
    2.0 * torch.tensor(class_counts, dtype=torch.float32)
)
class_weights = class_weights.to(DEVICE)

model = GGNNClassifier(hidden_dim=HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

best_val_f1 = -1.0
checkpoint_path = ARTIFACTS / "ggnn.pt"

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for x, edges, edge_types, graph_ids, labels in train_loader:
        x = x.to(DEVICE)
        edges = edges.to(DEVICE)
        edge_types = edge_types.to(DEVICE)
        graph_ids = graph_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x, edges, edge_types, graph_ids)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    val_metrics = evaluate(model, val_loader)
    average_loss = total_loss / max(len(train_loader), 1)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"loss={average_loss:.4f} | "
        f"val_accuracy={val_metrics['accuracy']:.4f} | "
        f"val_f1={val_metrics['f1']:.4f}"
    )

    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": 64,
                "hidden_dim": HIDDEN_DIM,
                "label_mapping": {
                    "0": "Non-vulnerable",
                    "1": "Vulnerable",
                },
            },
            checkpoint_path,
        )

print("\nBest validation F1:", round(best_val_f1, 4))
print("Saved checkpoint:", checkpoint_path)

# Reload the best checkpoint before final test evaluation.
checkpoint = torch.load(
    checkpoint_path, map_location=DEVICE, weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])

test_metrics = evaluate(model, test_loader)

with open(ARTIFACTS / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(test_metrics, file, indent=2)

print("\nFinal test metrics:")
print(json.dumps(test_metrics, indent=2))
print("\nMetrics saved to artifacts/metrics.json")
