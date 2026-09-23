from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from likely import Likely


def main() -> None:
    load_dotenv()

    client = TypeSafeClient()
    likely = Likely(client)

    state = "The building is on fire and people are trapped inside."
    question = "Does this describe an urgent situation?"

    score = likely(question, state)
    print(f"P({question!r} | {state!r}) = {score:.2f}")


if __name__ == "__main__":
    main()
