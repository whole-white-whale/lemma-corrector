# 🧬 lemma-corrector

**Treating text corpora like sequenced genomes to fix lemmatization artifacts.**

Standard NLP lemmatizers often propagate spelling errors, resulting in corrupted, non-existent lemmas (e.g., _happi_ instead of _happy_, or _runn_ instead of _run_). `lemma-corrector` bypasses traditional rigid dictionary lookups by applying **graph-based error-correction algorithms** inspired by bioinformatics to your text corpus.

By treating lemmas as nodes in a similarity graph, this library uses **Louvain community detection** and **PageRank consensus** to identify and repair misspelled lemmas without relying on pre-defined dictionaries.

## ✨ Features

- **Graph-Based Correction**: Maps lemmas into a NetworkX graph based on Damerau-Levenshtein edit distance.
- **Smart Community Detection**: Uses the Louvain algorithm to cluster morphologically similar lemmas together.
- **Consensus Leader Election**: Identifies the correct spelling by finding the node that is _both_ the most frequent and the most central (PageRank) in its community.
- **Vocabulary Guardrails**: Optionally accepts a set of known correct words to prevent over-correcting valid, domain-specific terminology.
- **Length-Aware Weighting**: Edge weights scale with string length, recognizing that a single typo in a long word is statistically more likely to be an error than a single character difference between two short words.

## 🚀 Quick Start

```python
from collections import Counter
from lemma_corrector import LemmaCorrector

# 1. Create a bag of lemmata (e.g., from your NLP pipeline)
# Notice the misspelled/over-lemmatized words: "happi", "runn", "quikly"
bag_of_lemmata = Counter({
    "happy": 150,
    "happi": 12,      # Misspelling
    "run": 200,
    "runn": 8,        # Misspelling
    "quickly": 90,
    "quikly": 5,      # Misspelling
    "the": 1000,
    "a": 800,
})

# 2. Initialize the corrector
corrector = LemmaCorrector(bag_of_lemmata)

# 3. Get the corrections
corrections = corrector.get_corrections()
print(corrections)
# Output: {'happi': 'happy', 'runn': 'run', 'quikly': 'quickly'}
```

### Using a Known Vocabulary

If you have domain-specific jargon that looks like a typo but is actually correct, pass a `vocabulary` set to protect those words:

```python
# "foo" and "bar" are domain-specific variables, not typos
vocabulary = {"foo", "bar", "baz"}

corrector = LemmaCorrector(bag_of_lemmata, vocabulary=vocabulary)
```

## 🧠 How It Works

The algorithm operates in four distinct phases:

1. **Graph Construction**:
   Each unique lemma becomes a node, weighted by its normalized frequency (frequency divided by the max frequency in the corpus). Edges are drawn between lemmas that have a Damerau-Levenshtein distance of exactly 1. Edge weights are calculated as `1.0 - (1.0 / max_length)`, meaning longer strings with a single edit get stronger connections.

2. **Community Detection**:
   The Louvain algorithm partitions the graph into communities. Lemmas that are structurally similar and connected are grouped together.

3. **Leader Election (Consensus)**:
   For each community, the algorithm looks for a "leader" (the correct spelling). To be chosen as the leader, a lemma must satisfy two conditions simultaneously:
   - It must be the **most frequent** lemma in the community (highest node weight).
   - It must be the **most central** lemma in the community (highest PageRank).
     _If the most frequent and most central lemmas are different, the community is deemed ambiguous and skipped._

4. **Vocabulary Validation**:
   If a vocabulary is provided, the algorithm enforces two rules:
   - Edges are never created between two known, correct words.
   - A community leader cannot be an unknown word if a known word already exists in that community.

## 📖 API Reference

### `LemmaCorrector`

The main class for correcting lemmas.

#### `__init__(bag_of_lemmata, vocabulary=None, epsilon=1.0e-6)`

- `bag_of_lemmata` (`Counter[str]`): A Counter mapping lemmas to their frequencies.
- `vocabulary` (`set[str] | None`): Optional set of known correct words to protect from correction.
- `epsilon` (`float`): Tolerance threshold for Louvain and PageRank algorithms.

#### `get_corrections() -> dict[str, str]`

Computes and returns a dictionary mapping misspelled lemmas to their corrected leaders.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📄 License

This project is licensed under the MIT License.
