from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from likely import Likely


def triage(likely: Likely, report: str) -> None:
    print(f"\nReport: {report!r}")

    urgency = likely("Does this describe an urgent, life-threatening situation?", report)
    print(f"  urgency = {urgency:.2f}")

    if urgency > 0.9:
        # Both questions below share `report` as state, so they're batched
        # into a single TypeSafe call alongside `urgency` above.
        fire = likely("Is fire involved?", report)
        medical = likely("Does this require immediate medical attention?", report)
        print(f"  fire = {fire:.2f}, medical = {medical:.2f}")
        print("  -> DISPATCH EMERGENCY SERVICES")
    else:
        followup = likely("Should this be escalated for follow-up within 24 hours?", report)
        print(f"  followup = {followup:.2f}")
        if followup > 0.5:
            print("  -> SCHEDULE FOLLOW-UP")
        else:
            print("  -> LOG AND CLOSE, no action needed")


def main() -> None:
    load_dotenv()

    client = TypeSafeClient()
    likely = Likely(client)

    reports = [
        "The building is on fire and people are trapped inside.",
        "Someone parked in the wrong spot in the parking lot again.",
        "A customer says their package arrived a day late and is asking for a refund.",
    ]
    for report in reports:
        triage(likely, report)


if __name__ == "__main__":
    main()
