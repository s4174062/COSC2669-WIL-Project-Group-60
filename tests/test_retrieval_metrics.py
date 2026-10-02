from retrieval_metrics import hit_at_k, labelled_source, ndcg_at_k, score_hits


def test_labelled_source_skips_out_of_scope():
    assert labelled_source("assessment_flexibility") == "assessment_flexibility"
    assert labelled_source("N/A (adversarial/out-of-scope test case)") is None


def test_hit_at_k():
    assert hit_at_k([0.0, 1.0, 0.0], 3) == 1.0
    assert hit_at_k([0.0, 0.0, 0.0], 3) == 0.0


def test_ndcg_perfect_and_miss():
    perfect = ndcg_at_k([1.0, 1.0, 1.0], 3)
    miss = ndcg_at_k([0.0, 0.0, 0.0], 3)
    first_only = ndcg_at_k([1.0, 0.0, 0.0], 3)
    assert perfect == 1.0
    assert miss == 0.0
    assert 0.0 < first_only < 1.0


def test_score_hits_uses_metadata_source():
    hits = [
        {"metadata": {"source": "leave_of_absence"}},
        {"metadata": {"source": "assessment_flexibility"}},
        {"metadata": {"source": "assessment_flexibility"}},
    ]
    result = score_hits(hits, "assessment_flexibility", k=3)
    assert result["hit_at_k"] == 1.0
    assert result["sources"][0] == "leave_of_absence"
    assert 0.0 < result["ndcg_at_k"] < 1.0
