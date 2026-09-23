import pytest

from likely import Likely


class FakeAnswer:
    def __init__(self, noul):
        self.noul = noul


class FakeResponse:
    def __init__(self, answers):
        self.answers = answers


class FakeScorer:
    """Stands in for TypeSafeClient: records every system_one call it receives."""

    def __init__(self, score=0.5):
        self.score = score
        self.calls = []

    def system_one(self, *, state, questions):
        self.calls.append((state, set(questions)))
        return FakeResponse({q: FakeAnswer(self.score) for q in questions})


@pytest.fixture
def scorer():
    return FakeScorer()


@pytest.fixture
def likely(scorer):
    return Likely(scorer)


def test_returns_scorer_result(likely, scorer):
    scorer.score = 0.75
    assert likely("Is it true?", "some state") == 0.75


def test_caches_result_without_refetching(likely, scorer):
    state = "cached state"
    likely("Q1", state)
    likely("Q1", state)
    assert len(scorer.calls) == 1


def test_batches_sibling_questions_in_one_call(likely, scorer):
    state = "shared state"
    likely("Question A", state)
    likely("Question B", state)
    assert len(scorer.calls) == 1
    call_state, call_questions = scorer.calls[0]
    assert call_state == state
    assert call_questions == {"Question A", "Question B"}


def test_prefetch_populates_cache(likely, scorer):
    state = "prefetched state"
    likely.prefetch(state, ["X", "Y"])
    assert len(scorer.calls) == 1
    assert likely("X", state) == scorer.score
    assert len(scorer.calls) == 1


def test_clear_forces_refetch(likely, scorer):
    state = "clearable"
    likely("Q", state)
    assert len(scorer.calls) == 1
    likely.clear()
    likely("Q", state)
    assert len(scorer.calls) == 2


def test_strips_whitespace_from_question_and_state(likely, scorer):
    likely("  Is it true?  ", "  some state  ")
    assert scorer.calls == [("some state", {"Is it true?"})]


@pytest.mark.parametrize("question,state", [("", "state"), ("   ", "state"), ("q", ""), ("q", "   ")])
def test_rejects_blank_question_or_state(likely, question, state):
    with pytest.raises(ValueError):
        likely(question, state)


def test_logs_each_call(likely, caplog):
    with caplog.at_level("DEBUG", logger="likely.api"):
        likely("Logged question", "logged state")
    assert any("Logged question" in record.getMessage() for record in caplog.records)


def test_batches_across_padded_literal_and_matching_variable_state(likely, scorer):
    state = "padded state"
    likely("Question A", state)
    likely("Question B", "  padded state  ")
    assert len(scorer.calls) == 1
    call_state, call_questions = scorer.calls[0]
    assert call_state == "padded state"
    assert call_questions == {"Question A", "Question B"}


def test_logs_truncate_long_state_content(likely, caplog):
    long_state = "Dear user, " + "x" * 500 + " sincerely, spammer"
    with caplog.at_level("DEBUG", logger="likely.api"):
        likely("Is this spam?", long_state)
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert long_state not in messages
    assert "chars)" in messages
