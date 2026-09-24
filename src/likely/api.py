"""Public API for the likely package."""

import ast
import logging
import sys
import linecache
import functools
import threading
from collections import OrderedDict
from dataclasses import dataclass
from itertools import islice
from types import FrameType, ModuleType
from typing import Callable

from typesafe_sdk import Noul, NoulAnswer, TypeSafeClient

logger = logging.getLogger(__name__)


def _preview(value: str, limit: int = 80) -> str:
    """Repr ``value`` for logging, truncated: state/question content can be long or sensitive."""
    if len(value) <= limit:
        return repr(value)
    return f"{value[:limit]!r}... ({len(value)} chars)"

@dataclass(frozen=True, order=True)
class _Noul:
    """A yes/no question plus optional descriptions of what counts as yes/no.

    Hashable and normalized (stripped; "" means "no criterion"), so it serves
    directly as a cache key and as a member of a batch. A future question type
    (e.g. Choice) only needs the same ``to_sdk``/``parse`` pair.
    """
    question: str
    yes: str = ""
    no: str = ""

    def __post_init__(self) -> None:
        for field in ("question", "yes", "no"):
            object.__setattr__(self, field, getattr(self, field).strip())
        if not self.question:
            raise ValueError("question must be a non-empty string")

    def to_sdk(self) -> Noul:
        if not (self.yes or self.no):
            return Noul(instructions=self.question)
        return Noul(
            instructions=self.question,
            criteria={"true": self.yes or None, "false": self.no or None},
        )

    def parse(self, answer: object) -> float:
        if not isinstance(answer, NoulAnswer):
            raise RuntimeError(f"missing or non-noul answer for {_preview(self.question)}")
        if not 0.0 <= answer.noul <= 1.0:
            raise RuntimeError(f"noul out of range for {_preview(self.question)}: {answer.noul}")
        return answer.noul


_Question = _Noul


# A string template from the source: literal text, or a placeholder given as
# (dotted name, f-string conversion, static format spec).
_Part = str | tuple[tuple[str, ...], int, str]
_Template = tuple[_Part, ...]

# Only values whose formatting can't run user code are rendered speculatively.
_SAFE_TYPES = (str, int, float, bool)


class _Unresolved(Exception):
    """A placeholder can't be rendered safely from the caller's frame."""


def _render(template: _Template, resolve: Callable[[tuple[str, ...]], object]) -> str:
    out: list[str] = []
    for part in template:
        if isinstance(part, str):
            out.append(part)
            continue
        path, conversion, spec = part
        value = resolve(path)
        if conversion == ord("r"):
            value = repr(value)
        elif conversion == ord("a"):
            value = ascii(value)
        elif conversion == ord("s"):
            value = str(value)
        out.append(format(value, spec))
    return "".join(out)


def _resolver(frame: FrameType) -> Callable[[tuple[str, ...]], object]:
    """Look up dotted names the way code in ``frame`` would, without running user code.

    A name the function binds locally is only ever read from its locals, so an
    unbound local never falls through to a global of the same name. Attribute
    steps are only taken through modules, read from their ``__dict__`` (so
    neither properties nor a module ``__getattr__`` can run).
    """
    code = frame.f_code
    local_names = {*code.co_varnames, *code.co_cellvars, *code.co_freevars}
    frame_locals = frame.f_locals

    def resolve(path: tuple[str, ...]) -> object:
        name, *attrs = path
        if name in frame_locals:
            value = frame_locals[name]
        elif name in local_names:
            raise _Unresolved(name)
        elif name in frame.f_globals:
            value = frame.f_globals[name]
        elif name in frame.f_builtins:
            value = frame.f_builtins[name]
        else:
            raise _Unresolved(name)
        for attr in attrs:
            if not isinstance(value, ModuleType) or attr not in vars(value):
                raise _Unresolved(".".join(path))
            value = vars(value)[attr]
        if type(value) not in _SAFE_TYPES:
            raise _Unresolved(".".join(path))
        return value

    return resolve


def _no_frame(path: tuple[str, ...]) -> object:
    raise _Unresolved(".".join(path))


@dataclass(frozen=True)
class _Spec:
    """A call site's question as written: literal text or an f-string template."""
    scope: tuple
    question: _Template
    yes: _Template = ()
    no: _Template = ()

    def render(self, resolve: Callable[[tuple[str, ...]], object]) -> _Question | None:
        """The question this call site would ask right now, or None if unknowable."""
        try:
            return _Noul(
                _render(self.question, resolve),
                _render(self.yes, resolve),
                _render(self.no, resolve),
            )
        except (_Unresolved, ValueError):
            return None


class _Indexer(ast.NodeVisitor):
    """Per-file index of calls shaped like f(text, <expr>[, yes=text][, no=text]).

    ``text`` is a string literal, or an f-string whose placeholders are plain
    names or module attributes (``{name}``, ``{config.NAME!r:>10}``).
    """
    def __init__(self) -> None:
        self.groups: dict[tuple, set[_Spec]] = {}   # (callee, group key) -> specs
        self.by_pos: dict[tuple, tuple] = {}      # call span -> ((callee, group key), scope)
        self.by_line: dict[int, list] = {}        # fallback for Python < 3.11
        self._scope = ("<module>",)

    def _visit_scope(self, node: ast.AST) -> None:
        prev = self._scope
        self._scope = (node.lineno, node.col_offset)
        self.generic_visit(node)
        self._scope = prev

    def visit_Call(self, n: ast.Call) -> None:
        self.generic_visit(n)
        if not (isinstance(n.func, (ast.Name, ast.Attribute)) and len(n.args) == 2):
            return
        x, y = n.args
        question = self._template(x)
        if question is None:
            return
        criteria = self._criteria(n.keywords)
        if criteria is None:
            return
        spec = _Spec(self._scope, question, **criteria)
        if isinstance(y, ast.Constant) and isinstance(y.value, str):
            gk = ("const", y.value.strip())
        else:
            # A non-string y (e.g. likely("q", 42)) isn't a valid call, since
            # state must be str; fall back to expr-identity, which just makes
            # such a call site match nothing else.
            gk = ("expr", self._scope, ast.dump(y))

        callee = ast.dump(n.func)
        entry = (callee, gk)

        self.groups.setdefault(entry, set()).add(spec)
        self.by_pos[(n.lineno, n.end_lineno, n.col_offset, n.end_col_offset)] = (entry, self._scope)
        self.by_line.setdefault(n.lineno, []).append((entry, self._scope))

    @classmethod
    def _criteria(cls, keywords: list[ast.keyword]) -> dict[str, _Template] | None:
        """Text ``yes=``/``no=`` keywords, or None if any keyword isn't one of those."""
        criteria: dict[str, _Template] = {}
        for kw in keywords:
            if kw.arg not in ("yes", "no"):
                return None
            if isinstance(kw.value, ast.Constant) and kw.value.value is None:
                continue
            template = cls._template(kw.value)
            if template is None:
                return None
            criteria[kw.arg] = template
        return criteria

    @classmethod
    def _template(cls, node: ast.expr) -> _Template | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return (node.value,)
        if not isinstance(node, ast.JoinedStr):
            return None
        parts: list[_Part] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(value.value)
                continue
            if not isinstance(value, ast.FormattedValue):
                return None
            path = cls._dotted(value.value)
            spec = cls._static_spec(value.format_spec)
            if path is None or spec is None:
                return None
            parts.append((path, value.conversion, spec))
        return tuple(parts)

    @classmethod
    def _dotted(cls, node: ast.expr) -> tuple[str, ...] | None:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            base = cls._dotted(node.value)
            return None if base is None else (*base, node.attr)
        return None

    @staticmethod
    def _static_spec(node: ast.expr | None) -> str | None:
        """A format spec with no placeholders of its own, e.g. the ``>10`` in ``{x:>10}``."""
        if node is None:
            return ""
        if isinstance(node, ast.JoinedStr) and all(
            isinstance(v, ast.Constant) for v in node.values
        ):
            return "".join(v.value for v in node.values)
        return None

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _visit_scope


@functools.lru_cache(maxsize=None)
def _index(filename: str) -> _Indexer:
    idx = _Indexer()
    try:
        idx.visit(ast.parse("".join(linecache.getlines(filename)), filename))
    except (SyntaxError, ValueError):
        pass                                   # no source -> no batching, still correct
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
        self._cache: OrderedDict[tuple[_Question, str], float] = OrderedDict()
        self._lock = threading.Lock()

    def __call__(self, question: str, state: str, *, yes: str | None = None, no: str | None = None) -> float:
        """Return how likely ``question`` is true given ``state``, in [0, 1].

        Args:
            question: A natural-language yes/no question.
            state: Context the question should be evaluated against.
            yes: Optional description of what counts as a "yes" answer.
            no: Optional description of what counts as a "no" answer.

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
        q = _Noul(question, yes or "", no or "")
        state = self._require_text(state, "state")

        key = (q, state)
        with self._lock:
            result = self._cache.get(key)
        cached = result is not None
        if not cached:
            siblings = self._siblings(sys._getframe(1), state)
            self._fetch(state, {q} | siblings, required={q})
            with self._lock:
                result = self._cache.get(key)
        if result is None:
            raise RuntimeError(f"no answer returned for question {_preview(q.question)}")
        logger.debug(
            "likely(question=%s, state=%s) -> %.4f (%s)",
            _preview(q.question), _preview(state), result, "cache hit" if cached else "fetched",
        )
        return result

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def _fetch(self, state: str, questions: set[_Question], required: set[_Question]) -> None:
        """Fetch answers for ``questions`` against ``state``.

        ``required`` marks which ones the caller actually asked for, as
        opposed to speculative siblings pulled in for batching. If a batch
        call fails, we retry its questions one at a time: a failure on a
        required question still propagates, but a failure on a
        merely-speculative one is logged and dropped, so a bad sibling can
        never break the caller's own request.
        """
        todo = self._pending(state, questions)
        for i in range(0, len(todo), self._max_batch_size):
            chunk = todo[i:i + self._max_batch_size]
            scores = self._fetch_chunk(state, chunk, required)
            self._store(state, scores)

    def _pending(self, state: str, questions: set[_Question]) -> list[_Question]:
        """The subset of ``questions`` not already cached for ``state``."""
        with self._lock:
            return sorted(q for q in questions if (q, state) not in self._cache)

    def _fetch_chunk(
        self, state: str, chunk: list[_Question], required: set[_Question],
    ) -> dict[_Question, float]:
        """Fetch one chunk, falling back to per-question retries if the batch call fails."""
        try:
            return self._request(state, chunk)
        except Exception:
            logger.warning(
                "batch fetch failed for %d question(s) on state=%s; retrying individually",
                len(chunk), _preview(state), exc_info=True,
            )
        return self._retry_individually(state, chunk, required)

    def _retry_individually(
        self, state: str, chunk: list[_Question], required: set[_Question],
    ) -> dict[_Question, float]:
        """Fetch each question in ``chunk`` on its own.

        A failure only propagates for a ``required`` question; a merely
        speculative sibling that fails is logged and dropped instead, so it
        can never take the caller's own question down with it.
        """
        scores: dict[_Question, float] = {}
        for q in chunk:
            try:
                scores.update(self._request(state, [q]))
            except Exception:
                if q in required:
                    raise
                logger.debug(
                    "dropping question %s: fetch failed and it was only a sibling",
                    _preview(q.question),
                )
        return scores

    def _store(self, state: str, scores: dict[_Question, float]) -> None:
        """Cache ``scores`` for ``state``, evicting the oldest entries past ``cache_size``."""
        with self._lock:
            for q, score in scores.items():
                self._cache[(q, state)] = score
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _request(self, state: str, questions: list[_Question]) -> dict[_Question, float]:
        logger.debug(
            "fetching %d question(s) for state=%s: %s",
            len(questions), _preview(state), ", ".join(_preview(q.question) for q in questions),
        )
        # Question ids are opaque to the model; the text can't be the id, since
        # the same text with different criteria is a different question.
        by_id = {f"q{i}": q for i, q in enumerate(questions)}
        response = self._scorer.system_one(
            state=state,
            questions={qid: q.to_sdk() for qid, q in by_id.items()},
        )
        return {q: q.parse(response.answers.get(qid)) for qid, q in by_id.items()}

    @staticmethod
    def _require_text(value: str, name: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _siblings(frame: FrameType, y: str) -> set[_Question]:
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
        entry, scope = hit
        callee, _ = entry
        empty = frozenset()
        # we merge on likely("...", state) where we join on calls where `state is the same expression or the same value as y
        specs = idx.groups.get(entry, empty) | idx.groups.get((callee, ("const", y)), empty)
        # f-string siblings are rendered from the caller's frame, so only those in
        # the caller's own scope can be; a stale value just wastes a sibling slot.
        resolve = _resolver(frame)
        rendered = (spec.render(resolve if spec.scope == scope else _no_frame) for spec in specs)
        return {q for q in rendered if q is not None}
