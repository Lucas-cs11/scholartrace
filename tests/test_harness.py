"""评测 harness 单元测试。"""
from src.schemas import PaperIdentity, RankLabel, RankResult, RunReport
from eval.harness import _validate_results, compute_p_r_f1, match_gold, summarize


def _paper(paper_id: str, title: str, doi: str = "") -> RankResult:
    return RankResult(paper=PaperIdentity(paper_id=paper_id, title=title, doi=doi), score=1.0)


# ------------------------------------------------------------------ S2 结构化输出校验
def test_validate_results_ok():
    results = [_paper("W1", "A"), _paper("W2", "B")]
    assert _validate_results(results) is True


def test_validate_results_empty_paper_id():
    results = [RankResult(paper=PaperIdentity(paper_id="", title="A"), score=0.5, label=RankLabel.HIGH)]
    assert _validate_results(results) is False


def test_validate_results_bad_score():
    results = [RankResult(paper=PaperIdentity(paper_id="W1", title="A"), score=1.5, label=RankLabel.HIGH)]
    assert _validate_results(results) is False


def test_compute_p_r_f1_perfect():
    gold = [{"doi:10.1/a"}, {"doi:10.1/b"}]
    pred = [_paper("W1", "A", "10.1/a"), _paper("W2", "B", "10.1/b")]
    m = compute_p_r_f1(pred, gold)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_compute_p_r_f1_partial():
    gold = [{"doi:10.1/a"}, {"doi:10.1/b"}]
    pred = [_paper("W1", "A", "10.1/a"), _paper("W9", "C", "10.9/c")]
    m = compute_p_r_f1(pred, gold)
    assert m["tp"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_gold_group_counts_once_even_with_multiple_identifiers():
    """同篇 gold 有 id+doi+title 三个标识，只算一篇。"""
    gold = [{"openalex:W1", "doi:10.1/a", "title:paper a"}]
    # 预测论文命中其中任一标识
    pred = [_paper("W1", "paper a", "10.1/a")]
    m = compute_p_r_f1(pred, gold)
    assert m["tp"] == 1
    assert m["recall"] == 1.0  # 不是 1/3


def test_match_gold_groups():
    ch = {"gold": [{"doi": "10.1/x", "title": "Some Title"}, {"openalex_id": "W2"}]}
    groups = match_gold(ch)
    assert len(groups) == 2
    assert "doi:10.1/x" in groups[0]
    assert "title:some title" in groups[0]
    assert "openalex:W2" in groups[1]


def test_match_gold_title_normalize():
    ch = {"gold": [{"title": "  Attention  Is All You Need  "}]}
    groups = match_gold(ch)
    assert "title:attention is all you need" in groups[0]


def test_match_gold_title_keep_letters_key():
    """基准用 keep_letters 归一化：容错标点/空白/unicode 差异。"""
    ch = {"gold": [{"title": "When Less is More: Investigating Data Pruning at   Scale"}]}
    groups = match_gold(ch)
    assert "title_n:whenlessismoreinvestigatingdatapruningatscale" in groups[0]


def test_compute_p_r_f1_keep_letters_tolerates_punct():
    """预测标题与 gold 标点/空白不同，keep_letters 键仍能命中。"""
    gold = match_gold({"gold": [{"title": "When Less is More: Investigating Data Pruning at   Scale"}]})
    pred = [_paper("W1", "When Less is More — Investigating Data Pruning at Scale")]
    m = compute_p_r_f1(pred, gold)
    assert m["tp"] == 1
    assert m["recall"] == 1.0


def test_summarize_empty():
    assert summarize([]) == {}


def test_summarize_mean():
    r1 = RunReport(query_id="a", f1=0.5, precision=1.0, recall=0.3333, latency_ms=100)
    r2 = RunReport(query_id="b", f1=0.0, precision=0.0, recall=0.0, latency_ms=200)
    s = summarize([r1, r2])
    assert s["mean_f1"] == 0.25
    assert s["mean_latency_ms"] == 150.0
