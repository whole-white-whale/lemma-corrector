from collections import Counter

from lemma_corrector import LemmaCorrector


class TestLemmaCorrector:
    def test_get_corrections_empty(self) -> None:
        texts: list[str] = []

        bag_of_lemmata = get_bag_of_lemmata(texts)
        corrector = LemmaCorrector(bag_of_lemmata)

        assert corrector.get_corrections() == {}

    def test_get_corrections_singleton(self) -> None:
        texts = [
            "spam",
        ]

        bag_of_lemmata = get_bag_of_lemmata(texts)
        corrector = LemmaCorrector(bag_of_lemmata)

        assert corrector.get_corrections() == {}

    def test_get_corrections(self) -> None:
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


def get_bag_of_lemmata(texts: list[str]) -> Counter[str]:
    return Counter(lemma.lower() for text in texts for lemma in text.split())
