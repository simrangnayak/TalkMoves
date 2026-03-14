import matplotlib.pyplot as plt
import pandas as pd
import re

df = pd.read_csv("predictions_ncte.csv")

LABEL_COLS = ["Predicted_Offering_Math_Help", "Predicted_Successful_Uptake"]


def get_last_speaker(previous_context: str) -> str | None:
    """Return 'S' or 'T' for the last labeled utterance in previous_context, or None."""
    matches = re.findall(r"\[(S|T)\]", previous_context)
    print(f"matches: {matches}")
    return matches[-1] if matches else None


def ratio_labeled_one_responding_to_student(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each label column, compute the fraction of rows labeled 1 where the
    student utterance is a direct response to a student utterance (i.e., the
    last speaker in previous_context is [S]).

    Returns a DataFrame with columns:
      - total_ones: total rows labeled 1
      - ones_responding_to_student: rows labeled 1 where last previous speaker is [S]
      - ratio: ones_responding_to_student / total_ones
    """
    last_speakers = df["previous_context"].apply(get_last_speaker)
    responding_to_student = last_speakers == "S"

    rows = []
    for col in LABEL_COLS:
        labeled_one = df[col] == 1
        total_ones = labeled_one.sum()
        ones_responding_to_student = (labeled_one & responding_to_student).sum()
        ratio = (
            ones_responding_to_student / total_ones if total_ones > 0 else float("nan")
        )
        rows.append(
            {
                "label": col,
                "total_ones": total_ones,
                "ones_responding_to_student": ones_responding_to_student,
                "ratio": ratio,
            }
        )

    return pd.DataFrame(rows).set_index("label")


def plot_student_to_student_pies():
    """
    Plot two pie charts showing the fraction of student-to-student interactions
    for 'Offering Math Help' and 'Successful Uptake' (manually verified counts).
    """
    data = {
        "Offering Math Help": {"student_to_student": 6, "total": 32},
        "Successful Uptake": {"student_to_student": 1, "total": 5},
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    colors = ["#4C72B0", "#DD8452"]

    for ax, (label, counts) in zip(axes, data.items()):
        s2s = counts["student_to_student"]
        other = counts["total"] - s2s
        ax.pie(
            [s2s, other],
            labels=["Student-to-Student", "Teacher-to-Student"],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax.set_title(f"{label}\n(n={counts['total']})", fontsize=13)

    fig.suptitle("Student-to-Student vs. Teacher-to-Student Interactions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("student_to_student_pies.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    result = ratio_labeled_one_responding_to_student(df)
    print(result.to_string())
    plot_student_to_student_pies()
