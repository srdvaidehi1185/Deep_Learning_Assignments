# app.py
from pathlib import Path

import torch
from flask import Flask, render_template, request, jsonify

from graph_builder import build_graph
from model import GGNNClassifier
from vulnerability_analyzer import analyze_vulnerabilities

app = Flask(__name__)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = Path("artifacts/ggnn.pt")
model = None


def load_model():
    global model

    if not CHECKPOINT_PATH.exists():
        print(
            "Model checkpoint not found. "
            "Run python train.py first."
        )
        model = None
        return

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = GGNNClassifier(
        input_dim=checkpoint.get("input_dim", 64),
        hidden_dim=checkpoint.get("hidden_dim", 64),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully.")


def graph_to_tensors(graph):
    x = torch.tensor(
        graph["x"],
        dtype=torch.float32,
        device=DEVICE,
    )

    if graph["edge_index"]:
        edge_index = torch.tensor(
            graph["edge_index"],
            dtype=torch.long,
            device=DEVICE,
        ).t().contiguous()
    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
            device=DEVICE,
        )

    edge_type = torch.tensor(
        graph["edge_type"],
        dtype=torch.long,
        device=DEVICE,
    )

    graph_ids = torch.zeros(
        x.size(0),
        dtype=torch.long,
        device=DEVICE,
    )

    return x, edge_index, edge_type, graph_ids


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}
    code = payload.get("code", "")

    if not isinstance(code, str) or not code.strip():
        return jsonify({
            "error": "Please enter some C/C++ code."
        }), 400

    if len(code) > 50000:
        return jsonify({
            "error": "Please keep the code below 50,000 characters."
        }), 400

    # Feature 1 and 2:
    # Find suspicious source lines and explain why they may be risky.
    findings = analyze_vulnerabilities(code)

    # Return findings even if the trained model is missing.
    # The website can still display the static-analysis results.
    if model is None:
        return jsonify({
            "error": (
                "The trained model is unavailable. "
                "Run python train.py to enable the model prediction."
            ),
            "findings": findings,
            "finding_count": len(findings),
        }), 503

    try:
        graph = build_graph(code)

        x, edge_index, edge_type, graph_ids = graph_to_tensors(
            graph
        )

        with torch.no_grad():
            logits = model(
                x,
                edge_index,
                edge_type,
                graph_ids,
            )

            probabilities = torch.softmax(logits, dim=1)[0]

        vulnerable_score = float(probabilities[1].item())
        safe_score = float(probabilities[0].item())
        prediction = int(probabilities.argmax().item())

        return jsonify({
            "prediction": (
                "Vulnerable"
                if prediction == 1
                else "Non-vulnerable"
            ),
            "label": prediction,
            "vulnerability_score": round(
                vulnerable_score * 100, 2
            ),
            "non_vulnerability_score": round(
                safe_score * 100, 2
            ),
            "nodes": len(graph["x"]),
            "edges": len(graph["edge_type"]),
            "tokens": len(graph["tokens"]),
            "findings": findings,
            "finding_count": len(findings),
            "warning": (
                "The model predicts at function level. "
                "Line-level findings are heuristic checks, not proof "
                "that a line is exploitable or that unflagged code is safe."
            ),
        })

    except Exception:
        app.logger.exception("Code analysis failed")

        return jsonify({
            "error": (
                "Analysis failed. Check the server terminal "
                "for details."
            ),
            "findings": findings,
            "finding_count": len(findings),
        }), 500


if __name__ == "__main__":
    load_model()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
