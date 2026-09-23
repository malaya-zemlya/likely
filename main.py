from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from likely import Likely


def main() -> None:
    load_dotenv()

    client = TypeSafeClient()
    likely = Likely(client)

    state = "The building is on fire and people are trapped inside."
    print("state=", state)
    score = likely("Does this describe an urgent situation?", state)
    print(f"P(q1|state) = {score:.2f}")
    score = likely("Is anything on fire?", state)
    print(f"P(q2|state) = {score:.2f}")
    

if __name__ == "__main__":
    main()
