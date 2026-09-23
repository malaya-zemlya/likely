"""Public API for the likely package."""

import ast
import logging
import sys
import linecache
import functools
import threading
from itertools import islice
from types import FrameType
from typing import Iterable

from typesafe_sdk import Noul, TypeSafeClient

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

    def __init__(self, scorer: TypeSafeClient, max_batch_size: int = 256) -> None:
        self._scorer = scorer
        self._max_batch_size = max_batch_size
        self._cache: dict[tuple[str, str], float] = {}
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
        """
        question = self._require_text(question, "question")
        state = self._require_text(state, "state")

        key = (question, state)
        cached = key in self._cache
        if not cached:
            self._fetch(state, {question} | self._siblings(sys._getframe(1), state))
        result = self._cache[key]
        logger.debug(
            "likely(question=%s, state=%s) -> %.4f (%s)",
            _preview(question), _preview(state), result, "cache hit" if cached else "fetched",
        )
        return result

    def prefetch(self, y: str, xs: Iterable[str]) -> None:
        """Manual escape hatch for dynamic x's the index can't see."""
        self._fetch(y, set(xs))

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def _fetch(self, y: str, xs: set[str]) -> None:
        y = self._require_text(y, "state")
        normalized = {v.strip() for v in xs if v.strip()}
        with self._lock:
            todo = sorted(v for v in normalized if (v, y) not in self._cache)
            for i in range(0, len(todo), self._max_batch_size):
                chunk = todo[i:i + self._max_batch_size]

                logger.debug(
                    "fetching %d question(s) for state=%s: %s",
                    len(chunk), _preview(y), [_preview(v) for v in chunk],
                )
                response = self._scorer.system_one(
                    state=y,
                    questions={v: Noul(instructions=v) for v in chunk},
                )
                scores = {v: response.answers[v].noul for v in chunk}
                self._cache.update({(v, y): scores[v] for v in chunk})

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
