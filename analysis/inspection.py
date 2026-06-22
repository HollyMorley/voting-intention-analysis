import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from config import AGREE_SCALE
from helpers import attitude_columns

def repeated_serials(df: pd.DataFrame) -> pd.DataFrame:
    """One row per serial_number that appears more than once."""
    others = df.columns.drop("serial_number")
    counts = df["serial_number"].value_counts()
    repeated = counts[counts > 1]

    rows = []
    for serial, n in repeated.items():
        block = df[df["serial_number"] == serial]
        all_identical = len(block.drop_duplicates(subset=others)) == 1
        rows.append({
            "serial_number": serial,
            "n_duplicated_rows": n,
            "all_identical": all_identical,
        })
    return pd.DataFrame(rows)

def dont_know_by_item(df: pd.DataFrame) -> pd.DataFrame:
    """Per-item 'Don't know' count and percentage across the attitudinal items."""
    cols = attitude_columns(df)
    is_dk = df[cols] == "Don't know"
    return pd.DataFrame({
        "n_dont_know": is_dk.sum(),
        "%_dont_know": (is_dk.mean() * 100).round(1),
    }).rename_axis("item").sort_values("%_dont_know", ascending=False)

def respondents_by_dont_know_count(df: pd.DataFrame) -> pd.Series:
    """How many respondents gave each number of 'Don't know' answers."""
    cols = attitude_columns(df)
    n_dk = (df[cols] == "Don't know").sum(axis=1)

    n_items, n_resp = len(cols), len(df)
    n_any = int((n_dk >= 1).sum())
    n_all = int((n_dk == n_items).sum())
    print(f"{n_resp} respondents across {n_items} attitudinal items")
    print(f"  {n_any} ({n_any/n_resp:.1%}) gave at least one 'Don't know'")
    print(f"  {n_all} ({n_all/n_resp:.1%}) answered 'Don't know' to all {n_items} items")

    return (n_dk.value_counts()
                .sort_index(ascending=False)
                .rename_axis("n_dont_know")   # index: how many DKs a respondent gave
                .rename("n_respondents"))      # value: how many respondents

# def response_style(df: pd.DataFrame) -> pd.Series:
#     """Show whether respondents differ in how they use the attitudinal scale.
#
#     Maps the agree/disagree scale to -2..+2 ('Don't know' -> NaN), then summarises,
#     per respondent:
#       baseline lean : mean score   (do they generally agree or disagree?)
#       scale usage   : SD of scores  (do they use the extremes or hug the middle?)
#
#     Plots both distributions and returns the headline summary statistics. A strong
#     mean inter-item correlation would point to acquiescence; a weak one suggests the
#     positive lean is more about item wording than a respondent 'agree' tendency.
#     """
#     cols = attitude_columns(df)
#     num = df[cols].apply(lambda s: s.map(AGREE_SCALE))   # -2..+2, DK -> NaN
#
#     resp_mean = num.mean(axis=1)   # baseline lean
#     resp_std = num.std(axis=1)     # scale usage
#
#     fig, axes = plt.subplots(1, 2, figsize=(11, 4))
#     resp_mean.hist(bins=30, ax=axes[0])
#     axes[0].axvline(0, color="k", lw=1)
#     axes[0].set(title="Per-respondent mean (baseline lean)",
#                 xlabel="mean score (-2..+2)", ylabel="respondents")
#     resp_std.hist(bins=30, ax=axes[1])
#     axes[1].set(title="Per-respondent SD (scale usage)",
#                 xlabel="SD of scores", ylabel="respondents")
#     fig.tight_layout()


def rare_responses(df:pd.DataFrame, threshold:int=1) -> pd.DataFrame:
    """List every categorical value appearing < threshold times."""
    out = []
    for col in df.select_dtypes(include="str").columns:
        vc = df[col].value_counts()
        rare = vc[vc < threshold]  # e.g. singletons when threshold=2
        for value, n in rare.items():
            out.append({"column": col, "value": value, "count": n})
    return pd.DataFrame(out)

def response_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Count and percentage of each response option for a single question.

    Missing values are included as their own row (dropna=False) so nothing is hidden.
    Rows are ordered most-common first.
    """
    counts = df[column].value_counts(dropna=False)
    return pd.DataFrame({
        "n": counts,
        "pct": (counts / len(df) * 100).round(1),
    }).rename_axis("response")

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V: strength of association (0 = none, 1 = perfect) between two
    categorical variables, derived from the chi-square statistic of their crosstab."""
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    r, k = confusion.shape
    return np.sqrt((chi2 / n) / (min(r, k) - 1))

def association_with_target(df: pd.DataFrame, target_col: str, columns=None) -> pd.Series:
    """Cramér's V of each categorical column against the target, strongest first.

    Defaults to every string column except the target. Note: variables with many
    categories (e.g. income bands, goal importance) tend to score a little higher,
    so read the ranking as indicative rather than exact.
    """
    cols = columns or [c for c in df.select_dtypes(include="str").columns if c != target_col]
    scores = {c: cramers_v(df[c], df[target_col]) for c in cols}
    return pd.Series(scores, name="cramers_v").sort_values(ascending=False)

def attitude_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of the attitudinal items mapped to -2..+2 ('Don't know' -> NaN).

    Labels are shortened (the shared 'general_attitudes_' prefix is stripped) so the
    heatmap is readable. Reveals which attitudes move together (issue dimensions).
    """
    cols = attitude_columns(df)
    num = df[cols].apply(lambda s: s.map(AGREE_SCALE))
    num.columns = [c.replace("general_attitudes_", "") for c in cols]
    return num.corr()