from collections import Counter
from itertools import combinations

import networkx as nx

from rapidfuzz.distance import DamerauLevenshtein


class LemmaCorrector:
    bag_of_lemmata: Counter[str]

    def __init__(self, bag_of_lemmata: Counter[str]) -> None:
        self.bag_of_lemmata = bag_of_lemmata

    def get_graph(self) -> nx.Graph:
        graph = nx.Graph()

        for lemma in self.bag_of_lemmata:
            graph.add_node(lemma)

        for node_1, node_2 in combinations(graph.nodes, 2):
            if DamerauLevenshtein.distance(node_1, node_2) == 1:
                graph.add_edge(node_1, node_2)

        return graph
