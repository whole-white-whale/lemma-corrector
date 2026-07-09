from collections import Counter
from itertools import combinations

import networkx as nx

from rapidfuzz.distance import DamerauLevenshtein


class LemmaCorrector:
    bag_of_lemmata: Counter[str]
    epsilon: float

    def __init__(self, bag_of_lemmata: Counter[str], epsilon: float = 1.0e-6) -> None:
        self.bag_of_lemmata = bag_of_lemmata
        self.epsilon = epsilon

    def get_graph(self) -> nx.Graph:
        graph = nx.Graph()

        if len(self.bag_of_lemmata) == 0:
            return graph

        max_lemma_frequency = self.bag_of_lemmata.most_common(1)[0][1]

        for lemma in self.bag_of_lemmata:
            graph.add_node(
                lemma, weight=self.bag_of_lemmata[lemma] / max_lemma_frequency
            )

        for node_1, node_2 in combinations(graph.nodes, 2):
            if DamerauLevenshtein.distance(node_1, node_2) == 1:
                graph.add_edge(node_1, node_2)

        return graph

    def get_communities(self, graph: nx.Graph) -> list[set[str]]:
        return nx.community.louvain_communities(graph, threshold=self.epsilon)

    def get_leader(self, graph: nx.Graph, community: set[str]) -> str | None:
        leader_by_node_weight = self.get_leader_by_node_weight(graph, community)
        leader_by_rank = self.get_leader_by_rank(graph, community)

        if leader_by_node_weight != leader_by_rank:
            return None

        return leader_by_rank

    def get_leader_by_node_weight(self, graph: nx.Graph, community: set[str]) -> str:
        return max(
            graph.subgraph(community).nodes.items(), key=lambda x: x[1]["weight"]
        )[0]

    def get_leader_by_rank(self, graph: nx.Graph, community: set[str]) -> str:
        return max(
            nx.pagerank(graph.subgraph(community), tol=self.epsilon).items(),
            key=lambda x: x[1],
        )[0]
