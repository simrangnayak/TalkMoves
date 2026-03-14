import csv
import os
import re
from openai import OpenAI


def _load_dotenv():
    """Load .env from script dir or cwd so OPENAI_API_KEY can be set there."""
    for dirpath in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        path = os.path.join(dirpath, ".env")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        if key.lower().startswith("export "):
                            key = key[7:].strip()
                        value = value.strip().strip("'\"")
                        # set if missing or currently empty so .env always wins when empty
                        if key and (key not in os.environ or not os.environ[key]):
                            os.environ[key] = value
            break


_load_dotenv()

MODEL = os.environ.get("LLM_MODEL", "gpt-5-mini")
BASE_URL = os.environ.get("OPENAI_BASE_URL") 

SYSTEM_PROMPT = """You are annotating student utterances from math classroom transcripts.

Your task is to label the focal student utterance using EXACTLY TWO binary values (0 or 1), in this order:

1. OFFERING MATH HELP
2. SUCCESSFUL UPTAKE

Each value must be either 0 or 1. Multiple labels may be 1 at the same time.

--------------------------------
TRANSCRIPT FORMAT
--------------------------------
The context is formatted with each line as: (N) [S/T] <text>
- (N) is the turn number in chronological order.
- [S] indicates the speaker is a student; [T] indicates the speaker is a teacher.
- Multiple lines with the same (N) and same role ([S] or [T]) belong to the same individual speaker.
- Multiple lines with the same (N) but different role ([S] vs [T]) indicate different individuals contributing to the same chronological turn.

--------------------------------
LABEL DEFINITIONS
--------------------------------

1. OFFERING MATH HELP (1 or 0): Label 1 if the student is offering mathematical assistance or contributing a solution step intended to help another student understand or proceed.

This includes:
- Offering a solution step or explanation
- Suggesting what to do next in a solution
- Explaining reasoning to help another student
- Offering to help or walk someone through the problem
- Helping even if prompted by the teacher

This does NOT include:
- Merely stating an answer without helping context
- Responding only to the teacher without helping peers
- Asking questions

Key idea: The student is helping someone move forward mathematically.


2. SUCCESSFUL UPTAKE (1 or 0): Label 1 if the student directly engages with or responds to another student's prior idea, solution, or reasoning.

This includes:
- Agreeing or disagreeing with a classmate's idea
- Building on or modifying another student's solution
- Referencing or reacting to what another student just said
- Continuing a peer's reasoning or plan

This does NOT include:
- Starting a completely new idea unrelated to peers
- Only responding to the teacher
- Asking unrelated questions
- Simply stating an answer without helping context

Key idea: The student is interacting with or building on another student's contribution.


--------------------------------
EXAMPLE (OFFERING HELP)
--------------------------------

PRE-UTTERANCE CONTEXT:
(55) [T] Not yet?
(55) [T] It's your turn now Bs.
(55) [T] Bs, what would you add?
(55) [T] What would you add?
(55) [T] Go for it.
(56) [T] High five.
(56) [T] Tucker, what advice would you give Doug?

FOCAL STUDENT UTTERANCE:
(57) [S] One is like the headband probably fell off his head so many times he forgot, and two is that .

POST-UTTERANCE CONTEXT:
(58) [T] But it's cute, right?
(59) [S] Yes.
(59) [S] You changed it to halves when you did that, so it's not going to be 1/12, it's going to be 1/2 because you changed it to halves in that problem.
(60) [T] Because we're dividing by two, right?
(61) [S] Would you call it B?
(62) [T] Hayden, what would you say?
(63) [S] I would say, again, like the last one is .

The focal student (Tucker) offers advice to Doug in response to the teacher's prompt → label: 1,0.

--------------------------------
EXAMPLE (SUCCESSFUL UPTAKE)
--------------------------------

PRE-UTTERANCE CONTEXT:
(80) [S] So I counted the top row was six and the bottom which was seven or eight.
(80) [S] And then... so then I found that half of both of those, because each half was four and half of each of those and for six is three and for seven is 3.5, but for eight it was four.
(80) [S] So then, I times-ed it together and I got 12.
(81) [T] Okay.
(81) [T] So we'll have to see if there's seven or eight or not...
(81) [T] Yes Jude?
(82) [S] So I saw it was seven.

FOCAL STUDENT UTTERANCE:
(82) [S] So I got a 21.

POST-UTTERANCE CONTEXT:
(83) [S] Yeah, same.
(84) [S] I got seven.
(84) [S] I counted seven.
(84) [S] I could like, it could be a-
(85) [T] Like seven-
(86) [S] No seven going down.
(87) [T] Okay.

The focal student builds on Jude's answer (seven) by sharing their own result (21) → label: 0,1.


--------------------------------
ANNOTATION RULES
--------------------------------

• Use pre- and post-context to interpret the focal utterance.
• Focus on the communicative function of the utterance.
• Multiple labels can be 1 simultaneously.
• Do not output explanations.

--------------------------------
OUTPUT FORMAT
--------------------------------

Respond ONLY with two digits separated by commas:

Offering math help, Successful uptake

Examples:
1,0
0,1"""


def build_prompt(pre_context: str, utterance: str, post_context: str, turn) -> str:
    try:
        turn_str = str(int(float(turn)))
    except (ValueError, TypeError):
        turn_str = "nan"
    focal = f"({turn_str}) [S] {utterance}"
    return f"""PRE-UTTERANCE CONTEXT:
{pre_context}

FOCAL STUDENT UTTERANCE:
{focal}

POST-UTTERANCE CONTEXT:
{post_context}

Respond with two digits: Offering math help, Successful uptake"""


def parse_response(text: str) -> tuple[int, int]:
    match = re.search(r"(\d)\s*,\s*(\d)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    digits = re.findall(r"\d", text)
    if len(digits) >= 2:
        return int(digits[0]), int(digits[1])
    return 0, 0


def annotate_row(client: OpenAI, row: dict) -> tuple[int, int]:
    prompt = build_prompt(
        row["previous_context"],
        row["student_utterance"],
        row["subsequent_context"],
        row["turn"],
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content or ""
    return parse_response(text)


def main():
    input_path = os.environ.get("INPUT_CSV", "talk_moves_validation_set.csv")
    output_path = os.environ.get("OUTPUT_CSV", "llm_annotated_validation_set.csv")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Set it with: export OPENAI_API_KEY='your-key'"
        )

    kwargs = {"api_key": api_key}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL

    client = OpenAI(**kwargs)

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    new_cols = ["Offering Math Help", "Successful Uptake"]
    for col in new_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    for i, row in enumerate(rows):
        offering, uptake = annotate_row(client, row)
        row["Offering Math Help"] = offering
        row["Successful Uptake"] = uptake
        print(f"{i+1}/{len(rows)}: {offering},{uptake}")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
