from collections import Counter
from itertools import combinations

import networkx as nx

from rapidfuzz.distance import DamerauLevenshtein


class LemmaCorrector:
    """Corrects misspelled lemmas in a text corpus using graph-based community detection.

    This class treats lemmas as nodes in a graph, connecting them with
    weighted edges if their Damerau-Levenshtein distance is 1. It then uses
    the Louvain algorithm to cluster similar lemmas into communities. The
    correct spelling (the "leader") of each community is determined by finding
    consensus between the lemma with the highest normalized frequency and the
    lemma with the highest PageRank.

    Attributes:
        bag_of_lemmata: A counter mapping each lemma to its frequency in the corpus.
        epsilon: The tolerance threshold used for community detection and PageRank
            calculations.
    """

    bag_of_lemmata: Counter[str]
    vocabulary: set[str]
    epsilon: float

    def __init__(
        self,
        bag_of_lemmata: Counter[str],
        vocabulary: set[str] | None = None,
        epsilon: float = 1.0e-6,
    ) -> None:
        """Initializes the LemmaCorrector with corpus data and tolerance settings.

        Args:
            bag_of_lemmata: A Counter object containing the lemmas and their
                respective frequencies from the text corpus.
            epsilon: The tolerance threshold for the Louvain community detection
                and PageRank algorithms. Defaults to 1.0e-6.
        """

        self.bag_of_lemmata = bag_of_lemmata

        if vocabulary is None:
            self.vocabulary = set()

        else:
            self.vocabulary = vocabulary

        self.epsilon = epsilon

    def get_corrections(self) -> dict[str, str]:
        """Computes the spelling corrections for misspelled lemmas in the corpus.

        Builds a similarity graph, detects communities of related lemmas, and
        identifies the correct spelling for each community. Misspelled lemmas
        are mapped to their community's "leader" (the correct spelling).

        Returns:
            A dictionary mapping misspelled lemmas to their corrected leader
            lemmas. Lemmas that are already correct or belong to ambiguous
            communities are excluded.
        """

        corrections: dict[str, str] = {}

        graph = self.get_graph()

        for community in self.get_communities(graph):
            if len(community) == 1:
                continue

            leader = self.get_leader(graph, community)

            if leader is None:
                continue

            for node in community:
                if node != leader:
                    corrections[node] = leader

        return corrections

    def get_graph(self) -> nx.Graph:
        """Constructs a NetworkX graph of lemmas based on edit distance and frequency.

        Nodes represent unique lemmas, weighted by their normalized frequency
        (frequency divided by the maximum frequency in the corpus). Weighted
        edges are added between any two lemmas that have a Damerau-Levenshtein
        distance of exactly 1. The edge weight is proportional to the length of
        the longer lemma, reflecting that a single edit is a smaller relative
        change in longer strings.

        Returns:
            An undirected NetworkX graph where nodes are lemmas and edges
            represent single-edit-distance similarities. Returns an empty graph
            if the bag of lemmata is empty.
        """

        graph = nx.Graph()

        if len(self.bag_of_lemmata) == 0:
            return graph

        max_lemma_frequency = self.bag_of_lemmata.most_common(1)[0][1]

        for lemma in self.bag_of_lemmata:
            graph.add_node(
                lemma, weight=self.bag_of_lemmata[lemma] / max_lemma_frequency
            )

        for node_1, node_2 in combinations(graph.nodes, 2):
            edge_weight = self.get_edge_weight(node_1, node_2)

            if edge_weight > 0.0:
                graph.add_edge(node_1, node_2, weight=edge_weight)

        return graph

    def get_edge_weight(self, node_1: str, node_2: str) -> float:
        """Calculates the edge weight between two lemmas based on edit distance and length.

        Returns 0.0 if the Damerau-Levenshtein distance between the two lemmas
        is not exactly 1. Otherwise, computes a weight as
        ``1.0 - 1.0 / max(len(node_1), len(node_2))``. This means that a single
        edit between longer strings produces a higher weight (closer to 1.0),
        reflecting greater overall similarity.

        Args:
            node_1: The lemma string #1.
            node_2: The lemma string #2.

        Returns:
            A float representing the edge weight. Returns 0.0 if the lemmas
            are not within a single edit distance, otherwise a value in the
            open interval (0.0, 1.0).
        """

        if node_1 in self.vocabulary and node_2 in self.vocabulary:
            return 0.0

        if DamerauLevenshtein.distance(node_1, node_2) != 1:
            return 0.0

        return 1.0 - 1.0 / max(len(node_1), len(node_2))

    def get_communities(self, graph: nx.Graph) -> list[set[str]]:
        """Detects communities of similar lemmas using the Louvain algorithm.

        Args:
            graph: The NetworkX graph of lemmas generated by `get_graph()`.

        Returns:
            A list of sets, where each set contains the lemmas belonging to a
            specific community.
        """

        return nx.community.louvain_communities(graph, threshold=self.epsilon)

    def get_leader(self, graph: nx.Graph, community: set[str]) -> str | None:
        """Determines the correct spelling (leader) for a given community of lemmas.

        A valid leader must be both the most frequent lemma (by node weight)
        and the most central lemma (by PageRank) within the community. If the
        most frequent and most central lemmas are not the same, the community
        is considered ambiguous and no leader is returned.

        Args:
            graph: The full NetworkX graph of lemmas.
            community: A set of lemmas representing a single community.

        Returns:
            The string of the correct lemma if there is consensus between
            frequency and PageRank, otherwise None.
        """

        leader_by_node_weight = self.get_leader_by_node_weight(graph, community)
        leader_by_rank = self.get_leader_by_rank(graph, community)

        if leader_by_node_weight != leader_by_rank:
            return None

        if leader_by_rank not in self.vocabulary and any(
            node in self.vocabulary for node in community
        ):
            return None

        return leader_by_rank

    def get_leader_by_node_weight(self, graph: nx.Graph, community: set[str]) -> str:
        """Finds the lemma with the highest normalized frequency in a community.

        Args:
            graph: The full NetworkX graph of lemmas.
            community: A set of lemmas representing a single community.

        Returns:
            The lemma string with the highest node weight in the given community.
        """

        return max(
            graph.subgraph(community).nodes.items(), key=lambda x: x[1]["weight"]
        )[0]

    def get_leader_by_rank(self, graph: nx.Graph, community: set[str]) -> str:
        """Finds the lemma with the highest PageRank in a community.

        Args:
            graph: The full NetworkX graph of lemmas.
            community: A set of lemmas representing a single community.

        Returns:
            The lemma string with the highest PageRank score in the given
            community subgraph.
        """

        return max(
            nx.pagerank(graph.subgraph(community), tol=self.epsilon).items(),
            key=lambda x: x[1],
        )[0]
