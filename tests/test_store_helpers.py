from store import _keyword_scores, _rrf, _tokens


def test_tokens_splits_identifier_styles():
    assert _tokens("AuthManager.login") == ["auth", "manager", "login"]
    assert _tokens("store_symbols") == ["store", "symbols"]
    assert _tokens("_walk") == ["walk"]
    assert _tokens("how are retries handled") == ["how", "are", "retries", "handled"]


def _catalog(*names):
    return {str(i): {"meta": {"name": n}} for i, n in enumerate(names)}


def test_keyword_exact_match_scores_high():
    scores = _keyword_scores("how does store_symbols work", _catalog("store_symbols", "AuthManager.login"))
    assert scores == {"0": 6.0}  # 'store' + 'symbols', exact = 3 each


def test_keyword_partial_match_scores_low():
    scores = _keyword_scores("connection pooling", _catalog("connect_db"))
    assert scores == {"0": 1.0}  # 'connection' partially matches 'connect'


def test_keyword_drops_stopwords_and_short_tokens():
    assert _keyword_scores("how does the code get X", _catalog("handler")) == {}


def test_keyword_returns_empty_when_no_signal_terms():
    assert _keyword_scores("how does it work", _catalog("anything")) == {}


def test_rrf_rewards_agreement_across_rankings():
    fused = _rrf({"x": 1, "y": 2}, {"x": 1, "z": 2})
    assert fused["x"] > fused["y"]
    assert fused["x"] > fused["z"]


def test_rrf_uses_k_constant():
    fused = _rrf({"a": 1}, k=60)
    assert fused["a"] == 1 / 61
