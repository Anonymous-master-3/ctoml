import math

import pytest

from ctoml.evaluation import corpus_metrics, edit_distance


def test_edit_distance_substitution_deletion_insertion():
    assert edit_distance('abc', 'axc') == 1
    assert edit_distance('abc', 'ac') == 1
    assert edit_distance('abc', 'abbc') == 1
    assert edit_distance('', 'abc') == 3


def test_corpus_counts_instead_of_average_utterance_rates():
    result = corpus_metrics(['a', 'b c d'], ['', 'b c e'])
    assert result['wer'] == .5
    assert result['cer'] == .5
    assert result['reference_words'] == 4
    assert result['word_errors'] == 2
    assert result['letter_accuracy'] == .5


def test_letter_accuracy_can_be_negative_and_empty_policy():
    assert corpus_metrics(['a'], ['abcd'])['letter_accuracy'] == -2
    assert corpus_metrics([''], [''])['letter_accuracy'] == 1
    assert math.isinf(corpus_metrics([''], ['x'])['wer'])
    assert corpus_metrics([''], ['x'])['letter_accuracy'] == -math.inf
    assert corpus_metrics(['', 'a'], ['x', 'a'])['cer'] == 1
    with pytest.raises(ValueError, match='counts'):
        corpus_metrics(['a'], [])


def test_case_punctuation_preserved_spaces_excluded_for_characters():
    result = corpus_metrics(['A- b'], ['a-b'])
    assert result['reference_characters'] == 3
    assert result['cer'] == pytest.approx(1 / 3)
