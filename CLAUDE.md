# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`likely` wraps the TypeSafe System One API (`typesafe-sdk`, docs at https://docs.typesafe.ai, index at `/llms.txt`) so that
`likely("Is this urgent?", state, yes=..., no=...)` returns the probability, in [0, 1], that a yes/no question is true, using TypeSafe `Noul` questions. User code is meant to read as ordinary `if` branching on those probabilities (see `main.py`).

## Commands

Managed with `uv` (Python 3.13).

- Run all tests: `uv run pytest -v`
- Run one test: `uv run pytest -v tests/test_likely.py::test_batches_sibling_questions_in_one_call`
- Run the demo (calls the real API; needs `TYPESAFE_API_KEY` in `.env`): `uv run python main.py`

## Architecture (`src/likely/api.py`)

The key idea is **speculative batching based on the AST of the calling code**. The first uncached `likely(q, state)` call also fetches every *sibling* question, so a single TypeSafe `system_one` call answers them all, including questions on branches that end up not running.

Internally a question is a `_Noul` object: frozen and normalized (stripped, with `""` meaning no criterion), holding the question text plus optional `yes`/`no` criteria. It serves as both a cache key and a batch member, and it knows how to build its SDK question (`to_sdk`) and validate its answer (`parse`). The fetch/cache plumbing is typed against the `_Question` alias, so a new question type (such as a planned Choice) plugs in there. In a request, question ids are `q0`, `q1`, and so on, not the question text, because the same text with different criteria is a different question.

The flow:

1. `Likely.__call__` looks up the caller's frame with `sys._getframe(1)` and calls `_siblings`.
2. `_index(filename)` (cached with `lru_cache`) parses the caller's source file with `_Indexer`. The indexer records every call shaped like `f(text, <expr>)`, optionally with text `yes=`/`no=` keywords, as a `_Spec`, grouped by `(callee AST dump, group key)`. `text` is a string literal, or an f-string whose placeholders are plain names or dotted names (`{name}`, `{config.NAME!r:>10}`) with static format specs. A call with any other keyword, or with any other keyword value, is not indexed. The group key is the stripped string value when the state is a string literal. Otherwise it is `(enclosing function scope, AST dump of the state expression)`.
3. On 3.11+ the current call site is found through `co_positions()` and `f_lasti`. On older versions it falls back to line number and only works when the line has exactly one call. The siblings are the specs in the call's own group plus those in any group whose constant state equals the runtime `state`. Each is rendered into a `_Noul`. f-string placeholders are resolved by `_resolver` from the caller's frame, following Python's scoping: a name the function binds locally is read only from its locals, so an unbound local is skipped rather than falling back to a global. Attribute steps go only through modules, read from `vars(module)`, and only values of exact type `str`/`int`/`float`/`bool` are formatted, so rendering never runs user code. Templates from a different scope than the caller's are not rendered. A sibling that can't be rendered is skipped. A sibling rendered with a value that changes before its line runs is just a wasted speculative question, because the real call always uses the runtime string.
4. `_fetch` splits the uncached questions into chunks of `max_batch_size`. When a batch request fails, `_retry_individually` asks each question on its own. A failure on a *required* question (the one the caller asked) propagates. A failure on a speculative sibling is logged and dropped.
5. Answers go into an LRU `OrderedDict` keyed by `(question, state)` and bounded by `cache_size`, which is protected by `threading.Lock`.

Consequences to keep in mind:

- Batching works for literal strings and for f-strings over names/module attributes whose values are already bound when a sibling call runs. Other dynamic questions (arbitrary expressions in placeholders, `.format()`, strings built elsewhere) still work, but each one costs its own request.
- Questions and states are `.strip()`ped everywhere, and blank values raise `ValueError`.
- If the source can't be parsed or found, batching is silently skipped. The results stay correct, but each call makes its own request.

## Tests

`tests/test_likely.py` replaces `TypeSafeClient` with a `FakeScorer` that records every `system_one(state=, questions=)` call. Batching tests assert on `scorer.calls`. Because siblings are found in the test file's own AST, the literal questions written in a test affect which ones get batched together.

## Python conventions

- Add type annotations to every function/method signature (params and return type), except test functions.
- Tests use pytest (`pytest -v`), fixtures over setup/teardown, files named `test_*.py`.
