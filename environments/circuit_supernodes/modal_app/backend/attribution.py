"""Circuit Tracer execution and graph export."""

from circuit_supernodes.runtime.graphs import MAX_FEATURE_NODES


class AttributionEngine:
    def __init__(self, model, node_threshold: float, edge_threshold: float):
        self.model = model
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold

    def _export_graph(self, graph) -> dict:
        """Export the ordinary pruned attribution graph."""

        import torch
        from circuit_tracer.graph import prune_graph
        from circuit_tracer.utils.create_graph_files import (
            build_model,
            create_nodes,
            create_used_nodes_and_edges,
        )
        from transformers import AutoTokenizer

        slug = "g"
        graph.to("cuda" if torch.cuda.is_available() else "cpu")
        node_mask, edge_mask, cumulative_scores = (
            value.cpu()
            for value in prune_graph(graph, self.node_threshold, self.edge_threshold)
        )
        graph.to("cpu")

        tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)
        nodes = create_nodes(graph, node_mask, tokenizer, cumulative_scores)
        used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
        model = build_model(
            graph,
            used_nodes,
            used_edges,
            slug,
            graph.scan,
            self.node_threshold,
            tokenizer,
        )
        return model.model_dump(mode="json")

    def attribute_graph(self, prompt: str):
        """Attribute an ordinary model execution."""

        import torch
        from circuit_tracer import attribute

        try:
            graph = attribute(
                prompt=prompt,
                model=self.model,
                max_n_logits=5,
                desired_logit_prob=0.95,
                max_feature_nodes=MAX_FEATURE_NODES,
                batch_size=8,
                offload=None,
                verbose=True,
            )
            return self._export_graph(graph)
        finally:
            if "graph" in locals():
                del graph
            torch.cuda.empty_cache()
