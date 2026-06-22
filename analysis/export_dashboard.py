"""Export pre-computed JSON for the static dashboard (docs/data/*.json).

Everything interactive in the dashboard is DISPLAY-ONLY: all modelling happens
here, in Python, and is frozen into JSON. Re-run this whenever the analysis
changes:

    python analysis/export_dashboard.py

It reproduces the analysis2.ipynb pipeline (clean -> encode -> curated value map
-> curated random forest) and writes four files:

    docs/data/respondents.json  one row per respondent: demographics, the two
                                value axes (econ, natint), actual vote, the model's
                                honest (out-of-fold) predicted vote + class
                                probabilities, and a `small` flag (SNP / Plaid Cymru
                                / Rather not say) so the dashboard can show/hide them.
    docs/data/importance.json   what the model leans on, computed TWICE: for the
                                full 9-class target ("all") and for main parties
                                only ("main"). Grouped permutation importance
                                (unique contribution) + univariate standalone power.
    docs/data/model_eval.json   held-out accuracy vs baselines, bootstrap CIs,
                                per-class recall and confusion matrix, again for
                                both "all" and "main".
    docs/data/meta.json         party colours, class lists, axis wording, the
                                items behind each axis -> drives the explainer text.

NB on the "all" vs "main" toggle: it is NOT two different models. The per-
respondent predictions come from one single 9-class forest (section 2 below).
Only importance/eval are recomputed, and even then it is the SAME pipeline and
features re-fit on a different set of rows (main = the 3 tiny classes dropped).
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from scipy.stats import ttest_1samp

from config import RAW_DATA_PATH, TARGET_COL, ECON_AXIS, NATINT_AXIS, BACKGROUND_GROUPS, ATTITUDE_PREFIX
from encode import INCOME_COL, POLITICAL_INTEREST_COL
import clean, encode, model
from helpers import pretty_labels, stars

OUT = Path(__file__).resolve().parent.parent / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# The curated feature set kept after the ablation comparison in analysis2.ipynb.
CHOSEN = "demographics+background+political+values+style+priorities [curated]"

# Party / response colours (extends the figure palette to the undecided classes).
PARTY_COLOURS = {
    "Conservative":     "#0087DC",
    "Labour":           "#E4003B",
    "Liberal Democrat": "#FAA61A",
    "SNP":              "#E6C700",
    "Plaid Cymru":      "#3F8428",
    "Another party":    "#7D3C98",
    "Don't know":       "#9E9E9E",
    "Would not vote":   "#6D6D6D",
    "Rather not say":   "#BDBDBD",
}
DECIDED = ["Conservative", "Labour", "Liberal Democrat", "SNP",
           "Plaid Cymru", "Another party"]
UNDECIDED = ["Don't know", "Would not vote", "Rather not say"]
CLASS_ORDER = DECIDED + UNDECIDED

# The three categories the user is unsure about showing: two tiny regional parties
# plus the non-answer. "Main parties only" hides exactly these.
SMALL = ["SNP", "Plaid Cymru", "Rather not say"]

# clean_data column names for the profiling demographics (raw survey columns).
WORKING_COL    = "which_of_the_following_best_describes_your_current_working_status"
REGION_COL     = "which_one_of_these_regions_do_you_live_in"
EU_COL         = "eu_referendum_how_did_you_vote_in_the_eu_referendum_in_june_2016"
GOAL_COL       = "goal_importance_most_important"
WORK_ORG_COL   = "work_organisation_please_tell_us_which_type_of_organisation_you_do_or_did_work_for"
WCLASS_COL     = "general_attitudes_i_consider_myself_working_class"
CARE_COL       = "which_of_the_following_statements_apply_to_you_i_care_for_a_child_with_a_serious_health_issue_disability"
TRANSPORT_COL  = "which_of_the_following_statements_apply_to_you_i_take_public_transport_regularly"


def r(x, n=3):
    """Round, turning NaN/None into None so the JSON stays valid for the browser."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), n)


def ci95(vals):
    """Analytic 95% CI of the mean: mean +/- 1.96 * sd / sqrt(n).

    Summarises ALL folds (unlike a 2.5/97.5 percentile of ~10 values, which is
    pinned to the two most extreme folds). Bar excluding 0 then lines up with the
    one-sample t-test star. NaN folds (a group absent in that split) are dropped.
    """
    v = pd.Series(vals).dropna().to_numpy()
    m = float(v.mean())
    if len(v) < 2:
        return m, m
    se = v.std(ddof=1) / np.sqrt(len(v))
    return m - 1.96 * se, m + 1.96 * se


# ---------------------------------------------------------------------------
# 1. Reproduce the notebook pipeline (clean -> encode -> features -> value map)
# ---------------------------------------------------------------------------
raw = pd.read_csv(RAW_DATA_PATH, encoding="utf-8")
cols_to_remove = ["which_of_the_following_statements_apply_to_you_dont_know"]
cols_with_missing = [
    "do_you_work_in",
    "eu_referendum_how_did_you_vote_in_the_eu_referendum_in_june_2016",
    "which_of_the_following_statements_apply_to_you_i_own_more_than_one_property",
]
clean_data = (raw
              .pipe(clean.drop_duplicate_records)
              .pipe(clean.remove_items, cols_to_remove)
              .pipe(clean.fill_missing_as_category, cols_with_missing, "No answer")
              .pipe(clean.fix_encoding_errors))

encoded = (clean_data
           .pipe(encode.engineer_attitudes)
           .pipe(encode.scale_political_interest)
           .pipe(encode.encode_income)
           .pipe(encode.binarise_yes_no))

X_all = encode.build_features(encoded, attitudes="normalised")
y_all = encoded[TARGET_COL]

value_map = encode.value_map_scores(X_all)               # columns: econ, natint
strength = encoded["opinion_strength"].reindex(X_all.index).fillna(0.0)

# Bin opinion strength into quartile labels for the demographic chart.
opinion_spread = pd.qcut(
    strength, q=4,
    labels=["Low (Q1)", "Medium-low (Q2)", "Medium-high (Q3)", "High (Q4)"],
    duplicates="drop",
).astype(str).replace("nan", None)

X_cur = encode.curated_feature_sets(X_all)[CHOSEN]
print(f"rows={len(X_all)}  curated features={X_cur.shape[1]}  classes={y_all.nunique()}")

# ---------------------------------------------------------------------------
# 2. Honest per-respondent predictions (out-of-fold), from the FULL 9-class model.
#    Used for the "swayable voter" profiles: a latent lean for everyone, including
#    the undecided, without training-set optimism. One consistent model so the
#    probabilities are comparable whichever view the dashboard shows.
# ---------------------------------------------------------------------------
cv = StratifiedKFold(5, shuffle=True, random_state=0)
proba = cross_val_predict(model.make_rf_pipeline(), X_cur, y_all,
                          cv=cv, method="predict_proba", n_jobs=-1)
_main_clf = model.make_rf_pipeline().fit(X_cur, y_all)
classes = list(_main_clf.classes_)
proba_df = pd.DataFrame(proba, index=X_cur.index, columns=classes)
oof_pred = proba_df.idxmax(axis=1)
print(f"out-of-fold accuracy (all rows) = {accuracy_score(y_all, oof_pred):.3f}")

# ---------------------------------------------------------------------------
# 3. Per-respondent records
# ---------------------------------------------------------------------------
records = []
for idx in X_all.index:
    p = proba_df.loc[idx]
    records.append({
        "id": int(clean_data.loc[idx, "serial_number"]),
        "age": clean_data.loc[idx, "age"],
        "actual_age": r(clean_data.loc[idx, "actual_age"], 0),
        "gender": clean_data.loc[idx, "gender"],
        "region": clean_data.loc[idx, REGION_COL],
        "income": clean_data.loc[idx, INCOME_COL],
        "working_status": clean_data.loc[idx, WORKING_COL],
        "political_interest": clean_data.loc[idx, POLITICAL_INTEREST_COL],
        "eu_referendum": clean_data.loc[idx, EU_COL],
        "goal": clean_data.loc[idx, GOAL_COL],
        "work_organisation": clean_data.loc[idx, WORK_ORG_COL],
        "working_class": clean_data.loc[idx, WCLASS_COL],
        # checkbox columns: NaN means the box was not ticked → "No"
        "care_disabled_child": "No" if pd.isna(clean_data.loc[idx, CARE_COL]) else str(clean_data.loc[idx, CARE_COL]),
        "public_transport": "No" if pd.isna(clean_data.loc[idx, TRANSPORT_COL]) else str(clean_data.loc[idx, TRANSPORT_COL]),
        "opinion_spread": opinion_spread[idx],
        "econ": r(value_map.loc[idx, "econ"]),
        "natint": r(value_map.loc[idx, "natint"]),
        "strength": r(strength.loc[idx], 2),
        "vote": y_all.loc[idx],
        "pred": oof_pred.loc[idx],
        "small": bool(y_all.loc[idx] in SMALL),
        "proba": {c: r(p[c]) for c in classes},
    })

records = json.loads(pd.DataFrame(records).where(pd.notna, None).to_json(orient="records"))
(OUT / "respondents.json").write_text(json.dumps(records))
print(f"wrote respondents.json ({len(records)} rows, "
      f"{sum(rec['small'] for rec in records)} flagged small)")


# ---------------------------------------------------------------------------
# 4. Model drivers + held-out evaluation, computed for each "policy"
#    ('all' = 9 classes, 'main' = small parties dropped). Same pipeline,
#    same features, different rows -> the dashboard toggle swaps between the two.
# ---------------------------------------------------------------------------
def evaluate_policy(idx):
    y = y_all.loc[idx]
    Xc = X_cur.loc[idx]
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    y_tr, y_te = y_all.loc[tr], y_all.loc[te]

    # --- drivers (on the training split) ---
    X_imp = Xc.loc[tr]
    groups = encode.curated_groups(X_imp)
    perm = model.grouped_permutation_importance(
        model.make_rf_pipeline, X_imp, y_tr, groups,
        cv=10, n_repeats=10, score_fn=balanced_accuracy_score)
    feat_cols = [c for c in perm.columns if c != "_fold"]
    fold_means = perm.groupby("_fold")[feat_cols].mean()
    perm_p = {c: ttest_1samp(fold_means[c], 0, alternative="greater").pvalue for c in feat_cols}

    uni = model.univariate_score(
        model.make_rf_pipeline, X_imp, y_tr, groups, cv=10, score_fn=balanced_accuracy_score)
    chance = 1.0 / y_tr.nunique()
    uni_p = {c: ttest_1samp(uni[c].dropna(), chance, alternative="greater").pvalue for c in uni.columns}

    blocks = []
    for c in feat_cols:
        fm = fold_means[c]
        perm_lo, perm_hi = ci95(fm)             # 95% CI of the mean over the 10 folds
        uni_lo, uni_hi = ci95(uni[c])
        blocks.append({
            "key": c, "label": pretty_labels(c),
            "perm_mean": r(fm.mean()), "perm_lo": r(perm_lo), "perm_hi": r(perm_hi),
            "perm_sig": stars(perm_p[c]),
            "uni_mean": r(uni[c].mean()), "uni_lo": r(uni_lo), "uni_hi": r(uni_hi),
            "uni_sig": stars(uni_p[c]),
        })
    blocks.sort(key=lambda d: d["perm_mean"], reverse=True)
    importance = {"chance": r(chance), "blocks": blocks}

    # --- held-out evaluation (fit once, score the test set) ---
    clf = model.make_rf_pipeline().fit(Xc.loc[tr], y_tr)
    pred = clf.predict(Xc.loc[te])
    present = [c for c in CLASS_ORDER if c in set(y_te)]
    mf = DummyClassifier(strategy="most_frequent").fit(Xc.loc[tr], y_tr).predict(Xc.loc[te])
    strat = DummyClassifier(strategy="stratified", random_state=0).fit(Xc.loc[tr], y_tr).predict(Xc.loc[te])

    yt = y_te.to_numpy()
    rng = np.random.default_rng(0)
    gap = np.array([accuracy_score(yt[ix], pred[ix]) - accuracy_score(yt[ix], mf[ix])
                    for ix in (rng.integers(0, len(yt), len(yt)) for _ in range(10000))])

    rng = np.random.default_rng(0)
    def bm(ix):
        a, b = yt[ix], pred[ix]
        out = {"accuracy": accuracy_score(a, b), "balanced_acc": balanced_accuracy_score(a, b)}
        for c in present:
            m = a == c
            out[f"recall:{c}"] = (b[m] == c).mean() if m.any() else np.nan
        return out
    boot = pd.DataFrame([bm(rng.integers(0, len(yt), len(yt))) for _ in range(2000)])
    ci = boot.quantile([0.025, 0.975])
    cm = confusion_matrix(y_te, pred, labels=present, normalize="true")

    per_class = [{
        "party": c,
        "recall": r((pred[yt == c] == c).mean()),
        "recall_lo": r(ci.loc[0.025, f"recall:{c}"]),
        "recall_hi": r(ci.loc[0.975, f"recall:{c}"]),
        "support": int((yt == c).sum()),
    } for c in present]

    eval_dict = {
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_classes": int(y.nunique()),
        "accuracy": r(accuracy_score(y_te, pred)),
        "accuracy_lo": r(ci.loc[0.025, "accuracy"]), "accuracy_hi": r(ci.loc[0.975, "accuracy"]),
        "balanced_accuracy": r(balanced_accuracy_score(y_te, pred)),
        "baselines": {
            "always Labour (most frequent)": r(accuracy_score(y_te, mf)),
            "random by frequency (stratified)": r(accuracy_score(y_te, strat)),
        },
        "gap_vs_baseline": {"mean": r(gap.mean()), "lo": r(np.quantile(gap, 0.025)), "hi": r(np.quantile(gap, 0.975))},
        "per_class": per_class,
        "confusion": {"labels": present, "matrix": [[r(v) for v in row] for row in cm]},
    }
    return importance, eval_dict


policies = {
    "all":  X_all.index,
    "main": y_all[~y_all.isin(SMALL)].index,
}
importance_out, eval_out = {}, {}
for name, idx in policies.items():
    print(f"evaluating policy '{name}' ({len(idx)} rows, {y_all.loc[idx].nunique()} classes)...")
    importance_out[name], eval_out[name] = evaluate_policy(idx)
    print(f"  test acc={eval_out[name]['accuracy']}  baseline={eval_out[name]['baselines']}")

(OUT / "importance.json").write_text(json.dumps(importance_out))
(OUT / "model_eval.json").write_text(json.dumps(eval_out))
print("wrote importance.json and model_eval.json (policies: all, main)")

# ---------------------------------------------------------------------------
# 5. Meta: colours, class lists, axis wording, the items behind each axis
# ---------------------------------------------------------------------------
meta = {
    "party_colours": PARTY_COLOURS,
    "decided": DECIDED,
    "undecided": UNDECIDED,
    "class_order": CLASS_ORDER,
    "small": SMALL,
    "n_respondents": int(len(X_all)),
    "axes": {
        "econ": {"label": "Economic", "low": "economic left", "high": "economic right",
                 "items_high": [s for s, w in ECON_AXIS.items() if w > 0],
                 "items_low": [s for s, w in ECON_AXIS.items() if w < 0]},
        "natint": {"label": "Nationalism–internationalism",
                   "low": "internationalist / open", "high": "nationalist / closed",
                   "items_high": [s for s, w in NATINT_AXIS.items() if w > 0],
                   "items_low": [s for s, w in NATINT_AXIS.items() if w < 0]},
    },
    "demographics": {
        "age":               "Age band",
        "region":            "Region",
        "eu_referendum":     "EU referendum vote",
        "political_interest": "Interest in politics",
        "work_organisation": "Work organisation",
        "goal":              "Top global goal",
        "working_class":     "Working class identity (socioeconomic)",
        "care_disabled_child": "Care for disabled child (vulnerability)",
        "public_transport":  "Uses public transport (cosmopolitan)",
        "opinion_spread":    "Opinion spread (response style)",
    },
}
(OUT / "meta.json").write_text(json.dumps(meta))
print("wrote meta.json")

# ---------------------------------------------------------------------------
# 6. Interactive persona tool: browser-side inference using the SAME main
#    curated RF (_main_clf, 100 trees, trained on all 1,468 respondents).
#    The JS starts from a per-demographic median feature vector (all background /
#    demographic features fixed) and lets the user override only the five
#    political features (political interest, econ, natint, EU vote, top goal).
# ---------------------------------------------------------------------------

# Impute X_cur so demographic median vectors are NaN-free at browser load time.
_imp = SimpleImputer(strategy="median").fit(X_cur)
X_cur_imp = pd.DataFrame(
    _imp.transform(X_cur), columns=X_cur.columns, index=X_cur.index
)
persona_cols = list(X_cur.columns)

# Column-index map for the 4 user-adjustable features (EU vote is now a
# demographic dimension, not a virtual-panel control).
GOAL_PRE = GOAL_COL + "_"      # e.g. goal_importance_most_important_Stopping climate change

control_map = {
    "political_interest": int(persona_cols.index(POLITICAL_INTEREST_COL)),
    "econ":               int(persona_cols.index("econ")),
    "natint":             int(persona_cols.index("natint")),
    "goal": {
        col[len(GOAL_PRE):]: int(persona_cols.index(col))
        for col in persona_cols if col.startswith(GOAL_PRE)
    },
}

# Per-demographic median vectors keyed by "region|age|gender|work_org|eu_vote".
# Each dimension also supports "All" (no filter on that axis).
GENDER_COL_RAW = "gender"
AGE_BAND_COL   = "age"
AGE_ORDER      = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
ALL = "All"

demo_df = pd.DataFrame({
    "region":   clean_data.loc[X_cur.index, REGION_COL],
    "age":      clean_data.loc[X_cur.index, AGE_BAND_COL],
    "gender":   clean_data.loc[X_cur.index, GENDER_COL_RAW],
    "work_org": clean_data.loc[X_cur.index, WORK_ORG_COL],
    "eu_raw":   clean_data.loc[X_cur.index, EU_COL],
    "goal_raw": clean_data.loc[X_cur.index, GOAL_COL],
}, index=X_cur.index)

region_vals   = sorted(demo_df["region"].unique().tolist())
age_vals      = [a for a in AGE_ORDER if a in set(demo_df["age"])]
gender_vals   = sorted(demo_df["gender"].unique().tolist())
work_org_vals = sorted(demo_df["work_org"].unique().tolist())
eu_vals       = ["Remain", "Leave", "No answer"]   # fixed display order

demographics = {}
print("computing demographic medians (5-dim × All combinations)...")
for r, a, g, w, e in itertools.product(
    [ALL] + region_vals,
    [ALL] + age_vals,
    [ALL] + gender_vals,
    [ALL] + work_org_vals,
    [ALL] + eu_vals,
):
    mask = pd.Series(True, index=demo_df.index)
    if r != ALL: mask &= demo_df["region"]   == r
    if a != ALL: mask &= demo_df["age"]      == a
    if g != ALL: mask &= demo_df["gender"]   == g
    if w != ALL: mask &= demo_df["work_org"] == w
    if e != ALL: mask &= demo_df["eu_raw"]   == e

    grp_idx = demo_df.index[mask]
    n = int(len(grp_idx))
    if n == 0:
        continue

    key = f"{r}|{a}|{g}|{w}|{e}"
    med = X_cur_imp.loc[grp_idx].median()
    sub = demo_df.loc[grp_idx]
    demographics[key] = {
        "n":       n,
        "medians": [round(float(v), 4) for v in med],
        "ref": {
            "pol_int": round(float(X_cur_imp.loc[grp_idx, POLITICAL_INTEREST_COL].median()), 2),
            "econ":    round(float(X_cur_imp.loc[grp_idx, "econ"].median()), 4),
            "natint":  round(float(X_cur_imp.loc[grp_idx, "natint"].median()), 4),
            "goal":    sub["goal_raw"].mode()[0],
        },
    }

print(f"demographic combinations exported: {len(demographics)}")

# Dropdown/toggle option lists for the JS.
demo_options = {
    "region":   [ALL] + region_vals,
    "age":      [ALL] + age_vals,
    "gender":   [ALL] + gender_vals,
    "work_org": [ALL] + work_org_vals,
    "eu_vote":  [ALL] + eu_vals,
}

# Serialize the main RF trees (same 100-tree model used throughout section 2).
main_rf = _main_clf.named_steps["model"]
print(f"serialising {len(main_rf.estimators_)} trees ({len(persona_cols)} features)...")


def _serialize_tree(estimator):
    """Pack one DecisionTree into parallel arrays for JS traversal."""
    t = estimator.tree_
    values = []
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            v = t.value[i][0]
            s = float(v.sum())
            values.append([round(float(x / s), 4) for x in v])
        else:
            values.append(None)
    return {
        "feature":   t.feature.tolist(),
        "threshold": [round(float(x), 6) for x in t.threshold],
        "left":      t.children_left.tolist(),
        "right":     t.children_right.tolist(),
        "value":     values,
    }


persona_model_export = {
    "n_features":     len(persona_cols),
    "classes":        list(main_rf.classes_),
    "pol_int_labels": ["None at all", "Not very much", "Some", "Quite a lot", "A great deal"],
    "control":        control_map,
    "demo_options":   demo_options,
    "demographics":   demographics,
    "trees": [_serialize_tree(t) for t in main_rf.estimators_],
}

(OUT / "persona_model.json").write_text(json.dumps(persona_model_export))
print("wrote persona_model.json")

# ---------------------------------------------------------------------------
# Bundle all JSON into data.js so the page works when opened from disk (file://)
# ---------------------------------------------------------------------------
bundle = {k: json.loads((OUT / f"{k}.json").read_text())
          for k in ["respondents", "importance", "model_eval", "meta", "persona_model"]}
(OUT.parent / "data.js").write_text("window.DASHBOARD_DATA = " + json.dumps(bundle) + ";\n")
print("wrote data.js (inlined bundle for file:// use)\nDONE")