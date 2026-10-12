import torch
from torch import nn


class GGNNClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_relations=4):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.relation_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_relations)
        ])

        self.update_cell = nn.GRUCell(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index, edge_type, graph_ids):
        """
        x:          [number_of_nodes, input_dim]
        edge_index: [2, number_of_edges]
        edge_type:  [number_of_edges]
        graph_ids:  [number_of_nodes], graph membership per node
        """
        h = torch.relu(self.input_projection(x))

        for _ in range(3):
            aggregate = torch.zeros_like(h)

            if edge_index.numel() > 0:
                source = edge_index[0]
                target = edge_index[1]

                for relation, layer in enumerate(self.relation_layers):
                    mask = edge_type == relation

                    if mask.any():
                        messages = layer(h[source[mask]])
                        aggregate.index_add_(
                            0, target[mask], messages
                        )

            h = self.update_cell(aggregate, h)

        num_graphs = int(graph_ids.max().item()) + 1
        pooled = torch.zeros(
            num_graphs, h.size(1),
            device=h.device,
            dtype=h.dtype
        )
        counts = torch.zeros(
            num_graphs, 1,
            device=h.device,
            dtype=h.dtype
        )

        pooled.index_add_(0, graph_ids, h)
        counts.index_add_(
            0,
            graph_ids,
            torch.ones(
                graph_ids.size(0), 1,
                device=h.device,
                dtype=h.dtype
            )
        )

        pooled = pooled / counts.clamp(min=1.0)
        return self.classifier(pooled)
