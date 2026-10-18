"""Small regression checks for offline paper table calculations."""

from export_paper_workbook import complete_time, paired_rows, quality


def test_set_coverage_is_not_exact_match_or_precision():
    q = quality([{'weather', 'air', 'wrong'}, set()], [{'weather', 'air'}, {'weather'}])
    assert q['tp'] == 2 and q['fp'] == 1 and q['fn'] == 1
    assert q['precision'] == q['recall'] == q['f1'] == 2 / 3
    assert q['covered'] == .5
    assert q['exact'] == 0
    assert q['wrong_queries'] == 1


def test_empty_final_sets_are_not_zero_second_completions():
    assert complete_time({}, set()) is None
    assert complete_time({'weather': 1.0}, {'weather', 'air'}) is None
    assert complete_time({'weather': 1.0, 'air': 2.0}, {'weather', 'air'}) == 2.0


def test_pairing_preserves_failure_quadrants_and_negative_leads():
    result = paired_rows('data', 'A', 'B',
                         [(3.0, 2.0), (1.0, None), (None, 1.0), (None, None)],
                         'GT', 'time', 'note')
    assert result[5:10] == [4, 1, 1, 1, 1]
    assert result[16:19] == [-1.0, -1.0, -1.0]
    assert result[19:22] == [0, 0, 1]
