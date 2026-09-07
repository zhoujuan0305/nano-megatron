from scripts.analyze_nsys_overlap import intersection_length, merge_intervals


def test_merge_intervals_coalesces_overlap_and_adjacency():
    assert merge_intervals([(5, 8), (1, 3), (2, 5), (10, 10)]) == [(1, 8)]


def test_intersection_length_uses_interval_unions():
    communication = [(0, 10), (8, 15), (20, 25)]
    compute = [(2, 4), (6, 12), (22, 30)]
    assert intersection_length(communication, compute) == 11
