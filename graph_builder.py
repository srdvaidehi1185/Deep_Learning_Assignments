import hashlib
import re

import numpy as np

try:
    from tree_sitter_language_pack import get_parser
    PARSER = get_parser("c")
except Exception:
    PARSER = None

FEATURE_DIM = 64
MAX_NODES = 256

TOKEN_PATTERN = re.compile(
    r"[A-Za-z_]\w*|"
    r"\d+(?:\.\d+)?|"
    r"==|!=|<=|>=|->|&&|\|\||"
    r"\+\+|--|\+=|-=|\*=|/=|"
    r"[^\s]"
)

CONTROL_WORDS = {
    "if", "else", "for", "while", "switch",
    "case", "return", "break", "continue", "goto"
}


def token_vector(token):
    """Create a deterministic hashed vector for a token."""
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)

    digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()

    for i in range(8):
        position = int.from_bytes(
            digest[i * 4:(i + 1) * 4], "little"
        ) % FEATURE_DIM
        vector[position] += 1.0

    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm

    return vector.tolist()


def extract_tokens(code):
    """Extract token text and parent identifiers when parsing succeeds."""
    if PARSER is not None:
        try:
            tree = PARSER.parse(code.encode("utf-8", errors="replace"))
            root = tree.root_node
            leaves = []

            def walk(node):
                if not node.children:
                    text = code.encode(
                        "utf-8", errors="replace"
                    )[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="replace"
                    )

                    if text.strip():
                        leaves.append({
                            "text": text,
                            "parent": (
                                node.parent.type
                                if node.parent is not None
                                else "root"
                            ),
                            "type": node.type,
                        })
                    return

                for child in node.children:
                    walk(child)

            walk(root)

            if leaves:
                return leaves
        except Exception:
            pass

    # Fallback tokenizer if parsing fails.
    return [
        {"text": token, "parent": "fallback", "type": "token"}
        for token in TOKEN_PATTERN.findall(code)
    ]


def build_graph(code):
    tokens = extract_tokens(code)[:MAX_NODES]

    if not tokens:
        tokens = [{"text": "<empty>", "parent": "root", "type": "token"}]

    features = []
    for item in tokens:
        vector = token_vector(item["text"])

        # Add a small control-keyword signal to the first feature.
        if item["text"] in CONTROL_WORDS:
            vector[0] += 0.5

        features.append(vector)

    # Relation IDs:
    # 0 = AST, 1 = CFG, 2 = DFG, 3 = NCS
    edges = []
    seen = set()

    def add_edge(source, target, relation, bidirectional=False):
        pair = (source, target, relation)
        if pair not in seen:
            seen.add(pair)
            edges.append([source, target, relation])

        if bidirectional:
            reverse = (target, source, relation)
            if reverse not in seen:
                seen.add(reverse)
                edges.append([target, source, relation])

    # NCS: sequential tokens.
    for i in range(len(tokens) - 1):
        add_edge(i, i + 1, 3, bidirectional=True)

    # CFG approximation: connect sequential tokens and control keywords
    # to the following token.
    for i in range(len(tokens) - 1):
        add_edge(i, i + 1, 1, bidirectional=False)

        if tokens[i]["text"] in CONTROL_WORDS:
            add_edge(i, i + 1, 1, bidirectional=True)

    # AST approximation: adjacent leaves sharing a parent.
    parent_groups = {}
    for i, item in enumerate(tokens):
        parent_groups.setdefault(item["parent"], []).append(i)

    for indices in parent_groups.values():
        for a, b in zip(indices, indices[1:]):
            add_edge(a, b, 0, bidirectional=True)

    # DFG approximation: connect occurrences of the same identifier.
    identifier_positions = {}
    for i, item in enumerate(tokens):
        word = item["text"]
        if re.fullmatch(r"[A-Za-z_]\w*", word):
            identifier_positions.setdefault(word, []).append(i)

    for positions in identifier_positions.values():
        # Limit edges so a common identifier cannot create a huge graph.
        positions = positions[:12]
        for a, b in zip(positions, positions[1:]):
            add_edge(a, b, 2, bidirectional=True)

    edge_index = [[e[0], e[1]] for e in edges]
    edge_type = [e[2] for e in edges]

    return {
        "x": features,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "tokens": [item["text"] for item in tokens],
    }
