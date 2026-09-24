# likely

Branch on plain-English questions like you would on booleans.

```python
if likely("Is this report about a fire?", report) > 0.5:
    dispatch_fire_department()
```

`likely(question, state)` returns the probability, in [0, 1], that a yes/no
`question` is true about `state`. It's backed by
[TypeSafe System One](https://docs.typesafe.ai) `Noul` questions.

Your code stays ordinary `if` statements. Behind the scenes, `likely`
reads the source of the calling function and **batches every question it
can see into one API call**, including questions on branches that haven't run yet.

## Example

```python
import logging
import sys

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from likely import Likely

SERVICE_AREA = "the downtown campus"


def triage(likely: Likely, report: str) -> None:
    print(f"\nReport: {report!r}")

    if likely(
        "Does this describe an urgent, life-threatening situation?", report,
        yes="someone could be hurt or killed if nobody responds within the hour",
        no="an inconvenience, complaint, or property issue with no danger to people",
    ) > 0.9:
        if likely("Is fire involved?", report) > 0.5:
            print("  -> DISPATCH FIRE DEPARTMENT")
        if likely("Does this require immediate medical attention?", report) > 0.5:
            print("  -> DISPATCH AMBULANCE")
        if likely(f"Is this happening at {SERVICE_AREA}?", report) > 0.5:
            print("  -> ALERT CAMPUS SECURITY")
    elif likely("Should this be escalated for follow-up within 24 hours?", report) > 0.5:
        print("  -> SCHEDULE FOLLOW-UP")
    else:
        print("  -> LOG AND CLOSE, no action needed")


load_dotenv()
logging.basicConfig(stream=sys.stdout, format="  %(message)s")
logging.getLogger("likely").setLevel(logging.DEBUG)   # show requests and probabilities

likely = Likely(TypeSafeClient())
triage(likely, "The building is on fire and people are trapped inside.")
triage(likely, "Someone parked in the wrong spot in the parking lot again.")
```

Output (log lines abbreviated):

```
Report: 'The building is on fire and people are trapped inside.'
  fetching 5 question(s) for state='The building is on fire and people are trapped inside.': ...
  likely(question='Does this describe an urgent, life-threatening situation?', ...) -> 0.9900 (fetched)
  likely(question='Is fire involved?', ...) -> 0.9900 (cache hit)
  -> DISPATCH FIRE DEPARTMENT
  likely(question='Does this require immediate medical attention?', ...) -> 0.7400 (cache hit)
  -> DISPATCH AMBULANCE
  likely(question='Is this happening at the downtown campus?', ...) -> 0.2500 (cache hit)

Report: 'Someone parked in the wrong spot in the parking lot again.'
  fetching 5 question(s) for state='Someone parked in the wrong spot in the parking lot again.': ...
  likely(question='Does this describe an urgent, life-threatening situation?', ...) -> 0.0200 (fetched)
  likely(question='Should this be escalated for follow-up within 24 hours?', ...) -> 0.3900 (cache hit)
  -> LOG AND CLOSE, no action needed
```

Each report costs **one** request, even though up to five questions get asked
and different reports take different branches. The full runnable version is
[`main.py`](main.py).

## How batching works

The first uncached `likely(question, state)` call also fetches its
*siblings*, meaning the other `likely(...)` calls in the same function with the same
state expression (or the same literal state string). Their answers are cached,
so the later calls are free.

What can be batched:

- **String literals**, e.g. `likely("Is fire involved?", report)`.
- **f-strings whose values are already known**, e.g.
  `likely(f"Is this happening at {SERVICE_AREA}?", report)`. Placeholders can be
  plain names (locals, parameters, globals) or module attributes
  (`{config.SERVICE_AREA}`), with conversions and format specs
  (`{name!r}`, `{count:03d}`). They're filled in from the caller's variables
  when the batch is built, following Python's scoping rules.
- **`yes=` / `no=` criteria**, as literals or f-strings under the same rules.

Batching is speculative and never affects correctness:

- If a placeholder's value changes before its line runs, that guess is wasted
  and the real call makes its own request with the actual string.
- Placeholders are only filled in with `str`, `int`, `float` or `bool` values, and
  attributes are only followed through modules. Guessing ahead never runs
  your code, including `__format__`, properties and module `__getattr__`.
- Anything else still works but isn't batched: arbitrary expressions in
  placeholders, `.format()`, or strings built elsewhere. Each such call costs its own
  request.
- In a loop, the questions for the current item are batched together, so you get
  one request per iteration.

## API

```python
likely = Likely(client, max_batch_size=256, cache_size=10_000)

p = likely(question, state, yes=None, no=None)   # float in [0, 1]
likely.clear()                                   # drop cached answers
```

- `question`: a natural-language yes/no question.
- `state`: the text the question is about.
- `yes` / `no`: optional descriptions of what counts as a yes or no answer.
- Blank questions or states raise `ValueError`. If TypeSafe can't answer the
  question you asked, you get a `RuntimeError`. A failing speculative sibling is
  logged and dropped, and never breaks your call.
- Answers are cached per `(question, criteria, state)` in a thread-safe LRU.

## Setup

Requires Python 3.13 and a TypeSafe API key.

```sh
uv add git+https://github.com/malaya-zemlya/likely
echo 'TYPESAFE_API_KEY=...' > .env
```

## Development

```sh
uv run pytest -v          # tests use a fake client, no API key needed
uv run python main.py     # demo against the real API
```

## License

[MIT No Attribution](LICENSE): do whatever you like with it. Attribution isn't
required, but a link back is appreciated.
