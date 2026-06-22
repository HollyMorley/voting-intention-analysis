"""Turn the cleaned survey frame into a model-ready feature matrix (X) and target (y).

Pipeline order in the notebook:
    model_df = (clean_data
                .pipe(scale_political_interest)
                .pipe(encode_income)
                .pipe(binarise_yes_no))
    X_all = build_features(model_df, attitudes="raw")
    y_all = group_parties(model_df[TARGET_COL])
    decided = y_all.isin(DECIDED_PARTIES)
    X, y = X_all[decided], y_all[decided]          # train/evaluate on these
    X_dk  = X_all[y_all == "Don't know"]           # score later for latent lean
"""
import numpy as np
import pandas as pd

from config import (AGREE_SCALE, TARGET_COL, ATTITUDE_PREFIX, ECON_AXIS, NATINT_AXIS,
                    BACKGROUND_GROUPS, DEMOGRAPHIC_COLS, DEMOGRAPHIC_BLOCKS,
                    POLITICAL_COLS, POLITICAL_BLOCKS, ENGAGEMENT_COLS, ENGAGEMENT_BLOCKS,
                    PRIORITY_COLS, PRIORITY_BLOCKS)
from helpers import attitude_columns

# --- column names ---
POLITICAL_INTEREST_COL = "political_interest_how_much_interest_do_you_generally_have_in_what_is_going_on_in_politics"
INCOME_COL = "what_is_the_combined_annual_income_of_your_household_prior_to_tax_being_deducted"

COL_MAPPING_FNS = {
}

# --- ordinal scales ---
POLITICAL_INTEREST_SCALE = {
    "None at all": 0,
    "Not very much": 1,
    "Some": 2,
    "Quite a lot": 3,
    "A great deal": 4,
}

INCOME_ORDER = [                # ascending; 'Prefer not to answer' handled separately
    "Up to £7,000",
    "£7,001 to £14,000",
    "£14,001 to £21,000",
    "£21,001 to £28,000",
    "£28,001 to £34,000",
    "£34,001 to £41,000",
    "£41,001 to £48,000",
    "£48,001 to £55,000",
    "£55,001 to £62,000",
    "£62,001 to £69,000",
    "£69,001 to £76,000",
    "£76,001 to £83,000",
    "£83,001 or more",
]

# --- target grouping ---
PARTY_RELABEL = {"Another party": "Other", "SNP": "Other", "Plaid Cymru": "Other"}

# columns that are never model features
NON_FEATURES = ["serial_number", "age", TARGET_COL]


def engineer_attitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric, within-respondent normalised, and uncertainty features for attitudes.

    For each attitudinal item, adds:
      <item>__num : 5-point agree scale mapped to -2..+2 ('Don't know' -> NaN)
      <item>__z   : <item>__num standardised *within each respondent* (ipsatised),
                    removing individual response style (acquiescence / extremity)

    Plus per-respondent uncertainty / engagement features:
      dk_rate          : proportion of items answered 'Don't know'
      opinion_strength : mean |score| over *answered* items (0 = neutral, 2 = strong);
                         NaN for respondents who answered 'Don't know' to everything
    """
    df = df.copy()
    cols = attitude_columns(df)

    num = df[cols].apply(lambda s: s.map(AGREE_SCALE))

    row_mean = num.mean(axis=1)
    row_std = num.std(axis=1).replace(0, np.nan)
    z = num.sub(row_mean, axis=0).div(row_std, axis=0)
    df["opinion_mean"] = row_mean
    df["opinion_std"] = row_std

    is_dk = df[cols] == "Don't know"
    df["dk_rate"] = is_dk.mean(axis=1)
    df["opinion_strength"] = num.abs().mean(axis=1)

    df = df.drop(columns=cols)
    return pd.concat([df, num.add_suffix("__num"), z.add_suffix("__z")], axis=1)


def scale_political_interest(df: pd.DataFrame) -> pd.DataFrame:
    """Map the ordinal political-interest scale to 0 (None at all) .. 4 (A great deal)."""
    df = df.copy()
    df[POLITICAL_INTEREST_COL] = df[POLITICAL_INTEREST_COL].map(POLITICAL_INTEREST_SCALE)
    return df


def encode_income(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode income; 'Prefer not to answer' -> NaN plus a refused flag.

    Replaces the income column with:
      income_ordinal : 0 (lowest band) .. 12 (highest); NaN if refused
      income_refused : 1 if 'Prefer not to answer', else 0
    """
    df = df.copy()
    order = {band: i for i, band in enumerate(INCOME_ORDER)}
    df["income_ordinal"] = df[INCOME_COL].map(order)
    df["income_refused"] = (df[INCOME_COL] == "Prefer not to answer").astype(int)
    return df.drop(columns=[INCOME_COL])


def binarise_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    """Map every Yes/No column to 1/0. Columns with any other value are left untouched."""
    df = df.copy()
    for col in df.columns:
        values = set(df[col].dropna().unique())
        if values and values <= {"Yes", "No"}:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def group_parties(target: pd.Series) -> pd.Series:
    """Collapse SNP / Plaid Cymru / Another party into 'Other'; leave other labels as-is."""
    return target.replace(PARTY_RELABEL)


def axis_score(X: pd.DataFrame, weights: dict) -> pd.Series:
    """Mean signed within-respondent attitude z-score for one value-map axis.

    `weights` maps an attitude stem to +/-1 (see config.ECON_AXIS / NATINT_AXIS).
    Uses the `__z` (ipsatised) columns so the score is net of individual response
    style; NaN z-scores (respondents with no opinion variance) count as neutral.
    Requires X built with attitudes="normalised".
    """
    cols = {f"{ATTITUDE_PREFIX}_{stem}__z": w for stem, w in weights.items()}
    return sum(w * X[c].fillna(0) for c, w in cols.items()) / len(weights)


def value_map_scores(X: pd.DataFrame) -> pd.DataFrame:
    """Place every respondent on the curated (econ, natint) political values map."""
    return pd.DataFrame(
        {"econ": axis_score(X, ECON_AXIS), "natint": axis_score(X, NATINT_AXIS)},
        index=X.index,
    )


def resolve_columns(columns, names) -> list:
    """Map curated base names to the actual feature columns they produced.

    A name matches a column if it is identical (numeric/binary feature kept as-is)
    or a prefix of it (a categorical that `get_dummies` expanded, e.g. 'gender' ->
    gender_Male / gender_Female). Order-preserving and de-duplicated.
    """
    out = []
    for n in names:
        out += [c for c in columns if c == n or c.startswith(n)]
    return list(dict.fromkeys(out))


def curated_feature_sets(X: pd.DataFrame) -> dict:
    """Ablation ladder of feature matrices, from demographics up to full X_all.

    demographics : the curated demographic blocks (age, gender, region, income, work)
    +background, +political, +values, +style, +priorities : progressively add the
            hand-curated blocks (config.py). 'values' is the curated econ/natint value-map
            scores (which replace the 11 raw attitude items); 'style' adds the engineered
            response-style / engagement features (opinion_mean/std, dk_rate); 'priorities'
            adds the one-hot goal_importance question. The ladder builds from *who people
            are* up to *what they believe, how they answer, and what they care about*.
    full  : the untouched one-hot X_all (adds the raw per-item attitudes in place of the
            value-map axes, on top of the curated set)
    Returns {name: DataFrame}; feed each to model.cross_validate_model with the SAME y.
    """
    values = value_map_scores(X)
    bg = resolve_columns(X.columns, [c for grp in BACKGROUND_GROUPS.values() for c in grp])
    demo = resolve_columns(X.columns, DEMOGRAPHIC_COLS)
    pol = resolve_columns(X.columns, POLITICAL_COLS)
    eng = resolve_columns(X.columns, ENGAGEMENT_COLS)   # opinion_mean/std, dk_rate
    prio = resolve_columns(X.columns, PRIORITY_COLS)    # goal_importance one-hot
    demo_bg = list(dict.fromkeys(demo + bg))
    demo_bg_pol = list(dict.fromkeys(demo + bg + pol))
    curated_values = pd.concat([X[demo_bg_pol], values], axis=1)
    curated_style = pd.concat([curated_values, X[eng]], axis=1)
    return {
        "demographics":                      X[demo],
        "demographics+background":           X[demo_bg],
        "demographics+background+political": X[demo_bg_pol],
        "demographics+background+political+values [curated]": curated_values,
        "demographics+background+political+values+style [curated]": curated_style,
        "demographics+background+political+values+style+priorities [curated]":
                                             pd.concat([curated_style, X[prio]], axis=1),
        "full X_all":                        X,
    }


def curated_groups(X: pd.DataFrame) -> dict:
    """Importance groups following the curated taxonomy (config.py).

    Block-granularity companion to curated_feature_sets, for permutation /
    univariate importance (model.grouped_permutation_importance / univariate_score):
    the two value-map axes as single columns, each life-background sub-domain as one
    block, each demographic question as one block, and each political-background
    question as one block (its one-hot dummies kept together).
    Only columns present in X are kept, so pass the curated matrix
    (curated_feature_sets(...)[CHOSEN]) which carries the 'econ'/'natint' columns.
    Returns {label: [columns]}.
    """
    groups = {}
    for axis in ("econ", "natint"):
        if axis in X.columns:
            groups[axis] = [axis]
    for name, cols in BACKGROUND_GROUPS.items():
        present = resolve_columns(X.columns, cols)
        if present:
            groups[name] = present
    for name, cols in DEMOGRAPHIC_BLOCKS.items():
        present = resolve_columns(X.columns, cols)
        if present:
            groups[name] = present
    for name, cols in POLITICAL_BLOCKS.items():
        present = resolve_columns(X.columns, cols)
        if present:
            groups[name] = present
    for name, cols in ENGAGEMENT_BLOCKS.items():
        present = resolve_columns(X.columns, cols)
        if present:
            groups[name] = present
    for name, cols in PRIORITY_BLOCKS.items():
        present = resolve_columns(X.columns, cols)
        if present:
            groups[name] = present
    return groups


def build_features(df: pd.DataFrame, attitudes: str = "raw") -> pd.DataFrame:
    """One-hot encode into the feature matrix (all rows; id / age / target dropped).

    attitudes:
      "raw"        -> per-item __num scores + opinion_strength + dk_rate
      "normalised" -> within-respondent __z scores + opinion_mean/std + dk_rate
    Never keep both __num and __z (they are collinear).

    Built on ALL rows so decided and undecided subsets share identical columns.
    """
    df = df.copy()

    num_cols = [c for c in df.columns if c.endswith("__num")]
    z_cols = [c for c in df.columns if c.endswith("__z")]
    if attitudes == "raw":
        df = df.drop(columns=z_cols + ["opinion_mean", "opinion_std"])
    elif attitudes == "normalised":
        df = df.drop(columns=num_cols + ["opinion_strength"])
    else:
        raise ValueError("attitudes must be 'raw' or 'normalised'")

    df = df.drop(columns=[c for c in NON_FEATURES if c in df.columns])
    return pd.get_dummies(df, dtype=int)        # one-hot the remaining categorical columns
