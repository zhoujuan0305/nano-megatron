from nano_megatron.parallel.rank_generator import RankGenerator


def test_tp_groups_world8_tp2_dp2_pp2():
    # order tp-cp-dp-pp, cp=1 → same as tp-dp-pp
    rg = RankGenerator(tp=2, dp=2, pp=2, cp=1, order="tp-cp-dp-pp")
    assert rg.get_ranks("tp") == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]


def test_dp_groups_world8_tp2_dp2_pp2():
    rg = RankGenerator(tp=2, dp=2, pp=2, cp=1, order="tp-cp-dp-pp")
    assert rg.get_ranks("dp") == [
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
    ]


def test_pp_groups_world8_tp2_dp2_pp2():
    rg = RankGenerator(tp=2, dp=2, pp=2, cp=1, order="tp-cp-dp-pp")
    assert rg.get_ranks("pp") == [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]


def test_cp_groups_world8_tp2_cp2_dp2():
    rg = RankGenerator(tp=2, dp=2, pp=1, cp=2, order="tp-cp-dp-pp")
    assert rg.get_ranks("cp") == [
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
    ]


def test_dp_cp_groups_world8_tp2_cp2_dp2():
    rg = RankGenerator(tp=2, dp=2, pp=1, cp=2, order="tp-cp-dp-pp")
    assert rg.get_ranks("dp-cp") == [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
    ]


def test_encode_decode_roundtrip():
    rg = RankGenerator(tp=2, dp=2, pp=2, cp=1, order="tp-cp-dp-pp")
    for rank in range(8):
        parts = rg.decode(rank)
        assert rg.encode(parts) == rank


def test_world_size_property():
    rg = RankGenerator(tp=2, dp=2, pp=2, cp=1)
    assert rg.world_size == 8
