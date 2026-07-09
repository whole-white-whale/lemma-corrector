from collections import Counter

from lemma_corrector import LemmaCorrector


class TestLemmaCorrector:
    """Tests for the LemmaCorrector class, specifically the get_corrections method.

    This test suite verifies the behavior of the correction algorithm under
    various conditions, including empty inputs, single-item inputs,
    standard multi-word corpora with spelling errors, and vocabulary constraints.
    """

    def test_get_corrections_empty(self) -> None:
        """Tests that an empty corpus yields an empty corrections dictionary."""

        texts: list[str] = []

        bag_of_lemmata = get_bag_of_lemmata(texts)
        corrector = LemmaCorrector(bag_of_lemmata)

        assert corrector.get_corrections() == {}

    def test_get_corrections_singleton(self) -> None:
        """Tests that a corpus with only one unique lemma yields no corrections.

        Verifies that the algorithm does not generate false corrections when
        there is only a single node in the graph and no neighboring lemmas
        to correct to.
        """

        texts = [
            "spam",
        ]

        bag_of_lemmata = get_bag_of_lemmata(texts)
        corrector = LemmaCorrector(bag_of_lemmata)

        assert corrector.get_corrections() == {}

    def test_get_corrections(self) -> None:
        """Tests the standard correction workflow with a multi-word corpus.

        Verifies that a low-frequency lemma ('baz') that is exactly one edit
        distance away from a high-frequency lemma ('bar') is correctly mapped
        to 'bar'. It also ensures that isolated, high-frequency lemmas ('foo')
        remain untouched and do not trigger corrections.
        """

        texts = [
            "foo foo foo",
            "bar bar",
            "baz",
        ]

        bag_of_lemmata = get_bag_of_lemmata(texts)
        corrector = LemmaCorrector(bag_of_lemmata)

        assert corrector.get_corrections() == {
            "baz": "bar",
        }

    def test_get_corrections_vocabulary(self) -> None:
        """Tests that providing a complete vocabulary prevents any corrections.

        Verifies that when all lemmas in the corpus are present in the known
        vocabulary, the algorithm correctly identifies them as valid words and
        yields an empty corrections dictionary, ensuring no known words are
        incorrectly overwritten.
        """

        texts = [
            "foo foo foo",
            "bar bar",
            "baz",
        ]

        vocabulary = {
            "foo",
            "bar",
            "baz",
        }

        bag_of_lemmata = get_bag_of_lemmata(texts)
        corrector = LemmaCorrector(bag_of_lemmata, vocabulary)

        assert corrector.get_corrections() == {}


def get_bag_of_lemmata(texts: list[str]) -> Counter[str]:
    """Extracts and counts lowercased words from a list of texts.

    This helper function simulates a basic tokenization/lemmatization step
    by splitting texts on whitespace and lowercasing the results to create
    a frequency distribution.

    Args:
        texts: A list of raw text strings to process.

    Returns:
        A Counter object mapping each unique lowercased word to its total
        frequency across all provided texts.
    """

    return Counter(lemma.lower() for text in texts for lemma in text.split())
