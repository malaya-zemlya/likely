"""Public API for the likely package."""

import ast
import logging
import sys
import linecache
import functools
import threading
from collections import OrderedDict
from itertools import islice
from types import FrameType
from typing import Iterable

from typesafe_sdk import Noul, NoulAnswer, TypeSafeClient

logger = logging.getLogger(__name__)


def _preview(value: str, limit: int = 80) -> str:
    """Repr ``value`` for logging, truncated: state/question content can be long or sensitive."""
    if len(value) <= limit:
        return repr(value)
    return f"{value[:limit]!r}... ({len(value)} chars)"

class _Indexer(ast.NodeVisitor):
    """Per-file index of calls shaped like f("literal", <expr>)."""
    def __init__(self) -> None:
        self.groups: dict[tuple, set[str]] = {}   # (callee, group key) -> constant x's
        self.by_pos: dict[tuple, tuple] = {}      # call span -> (callee, group key)
        self.by_line: dict[int, list] = {}        # fallback for Python < 3.11
        self._scope = ("<module>",)

    def _visit_scope(self, node: ast.AST) -> None:
        prev = self._scope
        self._scope = (node.lineno, node.col_offset)
        self.generic_visit(node)
        self._scope = prev

    def visit_Call(self, n: ast.Call) -> None:
        self.generic_visit(n)
        if not (isinstance(n.func, (ast.Name, ast.Attribute)) and 
                len(n.args) == 2 and
                not n.keywords):
            return
        x, y = n.args
        if not (isinstance(x, ast.Constant) and isinstance(x.value, str)):
            return
        x_value = x.value.strip()
        if not x_value:
            return
        if isinstance(y, ast.Constant) and isinstance(y.value, str):
            gk = ("const", y.value.strip())
        else:
            # A non-string y (e.g. likely("q", 42)) isn't a valid call, since
            # state must be str; fall back to expr-identity, which just makes
            # such a call site match nothing else.
            gk = ("expr", self._scope, ast.dump(y))

        callee = ast.dump(n.func)
        entry = (callee, gk)

        self.groups.setdefault(entry, set()).add(x_value)
        self.by_pos[(n.lineno, n.end_lineno, n.col_offset, n.end_col_offset)] = entry
        self.by_line.setdefault(n.lineno, []).append(entry)

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _visit_scope


@functools.lru_cache(maxsize=None)
def _index(filename: str) -> _Indexer:
    idx = _Indexer()
    try:
        idx.visit(ast.parse("".join(linecache.getlines(filename)), filename))
    except (SyntaxError, ValueError):
        pass                                   # no source -> no prefetch, still correct
    return idx

class Likely:
    """Estimate how likely a question is true given some state, via TypeSafe."""

    def __init__(
        self,
        scorer: TypeSafeClient,
        max_batch_size: int = 256,
        cache_size: int = 10_000,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        self._scorer = scorer
        self._max_batch_size = max_batch_size
        self._cache_size = cache_size
        self._cache: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._lock = threading.Lock()

    def __call__(self, question: str, state: str) -> float:
        """Return how likely ``question`` is true given ``state``, in [0, 1].

        Args:
            question: A natural-language yes/no question.
            state: Context the question should be evaluated against.

        Returns:
            A float in [0.0, 1.0]: the probability of a "yes"/true answer.

        Raises:
            ValueError: If ``question`` or ``state`` is empty or all whitespace.
            RuntimeError: If TypeSafe couldn't answer ``question`` itself --
                e.g. it returned a missing/wrong-typed or out-of-range
                answer, or the answer was evicted from the cache by a
                concurrent fetch before it could be read back. A failure on
                a merely speculative sibling question never raises here.
        """
        question = self._require_text(question, "question")
        state = self._require_text(state, "state")

        key = (question, state)
        with self._lock:
            result = self._cache.get(key)
        cached = result is not None
        if not cached:
            siblings = self._siblings(sys._getframe(1), state)
            self._fetch(state, {question} | siblings, required={question})
            with self._lock:
                result = self._cache.get(key)
        if result is None:
            raise RuntimeError(f"no answer returned for question {_preview(question)}")
        logger.debug(
            "likely(question=%s, state=%s) -> %.4f (%s)",
            _preview(question), _preview(state), result, "cache hit" if cached else "fetched",
        )
        return result

    def prefetch(self, questions: Iterable[str], state: str) -> None:
        """Manual escape hatch for dynamic questions the AST indexer can't see.

        Argument order matches ``__call__(question, state)``.
        """
        if isinstance(questions, str):
            raise TypeError("questions must be an iterable of strings, not a single string")
        self._fetch(state, set(questions))

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def _fetch(self, state: str, questions: set[str], required: set[str] | None = None) -> None:
        """Fetch answers for ``questions`` against ``state``.

        ``required`` (defaulting to all of ``questions``) marks which ones the
        caller actually asked for, as opposed to speculative siblings pulled
        in for batching. If a batch call fails, we retry its questions one at
        a time: a failure on a required question still propagates, but a
        failure on a merely-speculative one is logged and dropped, so a bad
        sibling can never break the caller's own request.
        """
        state = self._require_text(state, "state")
        normalized = {q.strip() for q in questions if q.strip()}
        required = normalized if required is None else {q.strip() for q in required if q.strip()}

        todo = self._pending(state, normalized)
        for i in range(0, len(todo), self._max_batch_size):
            chunk = todo[i:i + self._max_batch_size]
            scores = self._fetch_chunk(state, chunk, required)
            self._store(state, scores)

    def _pending(self, state: str, questions: set[str]) -> list[str]:
        """The subset of ``questions`` not already cached for ``state``."""
        with self._lock:
            return sorted(q for q in questions if (q, state) not in self._cache)

    def _fetch_chunk(self, state: str, chunk: list[str], required: set[str]) -> dict[str, float]:
        """Fetch one chunk, falling back to per-question retries if the batch call fails."""
        try:
            return self._request(state, chunk)
        except Exception:
            logger.warning(
                "batch fetch failed for %d question(s) on state=%s; retrying individually",
                len(chunk), _preview(state), exc_info=True,
            )
        return self._retry_individually(state, chunk, required)

    def _retry_individually(self, state: str, chunk: list[str], required: set[str]) -> dict[str, float]:
        """Fetch each question in ``chunk`` on its own.

        A failure only propagates for a ``required`` question; a merely
        speculative sibling that fails is logged and dropped instead, so it
        can never take the caller's own question down with it.
        """
        scores: dict[str, float] = {}
        for q in chunk:
            try:
                scores.update(self._request(state, [q]))
            except Exception:
                if q in required:
                    raise
                logger.debug(
                    "dropping question %s: fetch failed and it was only a sibling",
                    _preview(q),
                )
        return scores

    def _store(self, state: str, scores: dict[str, float]) -> None:
        """Cache ``scores`` for ``state``, evicting the oldest entries past ``cache_size``."""
        with self._lock:
            for q, score in scores.items():
                self._cache[(q, state)] = score
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _request(self, state: str, questions: list[str]) -> dict[str, float]:
        logger.debug(
            "fetching %d question(s) for state=%s: %s",
            len(questions), _preview(state), [_preview(q) for q in questions],
        )
        response = self._scorer.system_one(
            state=state,
            questions={q: Noul(instructions=q) for q in questions},
        )
        scores: dict[str, float] = {}
        for q in questions:
            answer = response.answers.get(q)
            if not isinstance(answer, NoulAnswer):
                raise RuntimeError(f"missing or non-noul answer for {_preview(q)}")
            if not 0.0 <= answer.noul <= 1.0:
                raise RuntimeError(f"noul out of range for {_preview(q)}: {answer.noul}")
            scores[q] = answer.noul
        return scores

    @staticmethod
    def _require_text(value: str, name: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _siblings(frame: FrameType, y: str) -> set[str]:
        code = frame.f_code
        idx = _index(code.co_filename)
        if sys.version_info >= (3, 11):
            pos = next(islice(code.co_positions(), frame.f_lasti // 2, None), None)
            hit = idx.by_pos.get(pos)
        else:
            cands = idx.by_line.get(frame.f_lineno, [])
            hit = cands[0] if len(cands) == 1 else None
        if hit is None:
            return set()
        callee, _ = hit
        empty = frozenset()
        # we merge on likely("...", state) where we join on calls where `state is the same expression or the same value as y
        return idx.groups.get(hit, empty) | idx.groups.get((callee, ("const", y)), empty)
