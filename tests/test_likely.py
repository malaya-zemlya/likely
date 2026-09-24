import pytest
from typesafe_sdk import NoulAnswer

from likely import Likely


class FakeResponse:
    def __init__(self, answers):
        self.answers = answers


class FakeScorer:
    """Stands in for TypeSafeClient: records every system_one call it receives."""

    def __init__(self, score=0.5):
        self.score = score
        self.calls = []          # (state, {question text}) per call
        self.sent = []           # every Noul object sent, in order
        self.fail_on = set()

    def system_one(self, *, state, questions):
        texts = {q.instructions for q in questions.values()}
        self.calls.append((state, texts))
        self.sent.extend(questions.values())
        if self.fail_on & texts:
            raise RuntimeError("simulated TypeSafe failure")
        return FakeResponse({qid: NoulAnswer(noul=self.score) for qid in questions})


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
    likely.prefetch(["X", "Y"], state)
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


def test_prefetch_rejects_bare_string(likely):
    with pytest.raises(TypeError):
        likely.prefetch("not a list of questions", "some state")


@pytest.mark.parametrize("kwarg", ["max_batch_size", "cache_size"])
def test_rejects_non_positive_constructor_args(scorer, kwarg):
    with pytest.raises(ValueError):
        Likely(scorer, **{kwarg: 0})


def test_missing_answer_raises(scorer):
    class MissingAnswerScorer:
        def system_one(self, *, state, questions):
            return FakeResponse({})

    likely = Likely(MissingAnswerScorer())
    with pytest.raises(RuntimeError):
        likely("Q", "state")


def test_out_of_range_noul_raises(scorer):
    class OutOfRangeScorer:
        def system_one(self, *, state, questions):
            return FakeResponse({q: NoulAnswer(noul=1.5) for q in questions})

    likely = Likely(OutOfRangeScorer())
    with pytest.raises(RuntimeError):
        likely("Q", "state")


def test_cache_is_bounded(scorer):
    bounded = Likely(scorer, cache_size=2)
    bounded("Q1", "s1")
    bounded("Q2", "s2")
    bounded("Q3", "s3")
    assert len(scorer.calls) == 3
    bounded("Q1", "s1")  # evicted as the oldest entry -> real refetch
    assert len(scorer.calls) == 4


def test_bad_sibling_does_not_break_real_question(likely, scorer):
    state = "shared state for sibling isolation"
    scorer.fail_on = {"Bad sibling"}
    if False:
        likely("Bad sibling", state)  # never runs; only registers as an AST sibling
    result = likely("Good question", state)
    assert result == scorer.score
    calls_before = len(scorer.calls)
    with pytest.raises(RuntimeError):
        likely("Bad sibling", state)  # nothing was cached for it, so it's fetched (and fails) again
    assert len(scorer.calls) > calls_before


def test_required_question_failure_propagates(likely, scorer):
    scorer.fail_on = {"Broken question"}
    with pytest.raises(RuntimeError):
        likely("Broken question", "state for the broken-question test")


def test_sends_criteria(likely, scorer):
    likely("Is it urgent?", "criteria state", yes="  someone could get hurt  ")
    (noul,) = scorer.sent
    assert noul.criteria == {"true": "someone could get hurt", "false": None}


def test_omits_criteria_when_not_given(likely, scorer):
    likely("Is it urgent?", "no-criteria state")
    (noul,) = scorer.sent
    assert noul.criteria is None


def test_same_question_with_different_criteria_are_distinct(likely, scorer):
    state = "distinct criteria state"
    likely("Is it bad?", state, yes="it breaks the law")
    likely("Is it bad?", state, no="it is merely rude")
    assert len(scorer.calls) == 1
    assert len(scorer.sent) == 2
    assert {n.criteria["true"] for n in scorer.sent} == {"it breaks the law", None}


def test_non_literal_criteria_is_not_batched(likely, scorer):
    state = "dynamic criteria state"
    desc = "computed at runtime"
    likely("Plain question", state)
    likely("Dynamic question", state, yes=desc)
    assert scorer.calls == [
        (state, {"Plain question"}),
        (state, {"Dynamic question"}),
    ]
