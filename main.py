import logging
import sys

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from likely import Likely

# App config: f-string questions built from it are still batched, because the
# value is already known when the first question about a report is asked.
SERVICE_AREA = "the downtown campus"


def triage(likely: Likely, report: str) -> None:
    print(f"\nReport: {report!r}")

    if likely(
        "Does this describe an urgent, life-threatening situation?", report,
        yes="someone could be hurt or killed if nobody responds within the hour",
        no="an inconvenience, complaint, or property issue with no danger to people",
    ) > 0.9:
        # The questions below share `report` as state, so they're batched
        # into a single TypeSafe call alongside the urgency question above --
        # including the f-string, rendered with SERVICE_AREA ahead of time.
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


def main() -> None:
    load_dotenv()
    # likely logs every request it sends and every probability it returns.
    logging.basicConfig(stream=sys.stdout, format="  %(message)s")
    logging.getLogger("likely").setLevel(logging.DEBUG)

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
