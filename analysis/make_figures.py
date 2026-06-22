from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             ConfusionMatrixDisplay)
from sklearn.dummy import DummyClassifier
from scipy.stats import ttest_1samp, t as student_t

from config import RAW_DATA_PATH, TARGET_COL, AGREE_SCALE
import clean, encode, model, inspection
from encode import INCOME_ORDER, POLITICAL_INTEREST_SCALE
from helpers import pretty_labels, stars, attitude_columns, FRIENDLY, _shorten

FIG = Path(__file__).resolve().parent.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
                     "axes.titleweight": "bold", "savefig.bbox": "tight"})

PARTY_COLOURS = {
    "Conservative": "#0087DC", "Labour": "#E4003B", "Liberal Democrat": "#FAA61A",
    "SNP": "#E6C700", "Plaid Cymru": "#3F8428", "Another party": "#7D3C98",
    "Don't know": "#9E9E9E", "Would not vote": "#6D6D6D", "Rather not say": "#BDBDBD",
}
DECIDED = ["Conservative", "Labour", "Liberal Democrat", "SNP", "Plaid Cymru", "Another party"]
UNCOMMITTED = ["Don't know", "Would not vote", "Rather not say"]
ORDER = DECIDED + UNCOMMITTED


def save(fig, name, show=True):
    fig.savefig(FIG / name, dpi=400)
    print("wrote", name)
    # Render inline when called from a notebook; closing first would prevent display.
    try:
        from IPython import get_ipython
        from IPython.display import display
        if show and get_ipython() is not None:
            display(fig)
    except ImportError:
        pass
    plt.close(fig)


def plot_voting_intention(target):
    vc = target.value_counts().reindex(ORDER).dropna()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(range(len(vc)), vc.values,
            color=[PARTY_COLOURS[p] for p in vc.index])
    ax.set_yticks(range(len(vc)), vc.index)
    ax.invert_yaxis()
    for i, v in enumerate(vc.values):
        ax.text(v + 3, i, f"{v}  ({v/len(target):.0%})", va="center", fontsize=9)
    unc = target.isin(UNCOMMITTED).sum()
    ax.set(xlabel="respondents", title="Vote intention is imbalanced - and a third are uncommitted")
    ax.text(0.98, 0.05, f"{unc/len(target):.0%} uncommitted\n(grey bars)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#555",
            bbox=dict(boxstyle="round", fc="#f2f2f2", ec="none"))
    ax.set_xlim(0, vc.max() * 1.18)
    save(fig, "01_target.png")

def response_style(df: pd.DataFrame):
    """Show whether respondents differ in how they use the attitudinal scale.

    Maps the agree/disagree scale to -2..+2 ('Don't know' -> NaN), then summarises,
    per respondent:
      baseline lean : mean score   (do they generally agree or disagree?)
      consistency   : SD of scores  (do they answer the 11 items similarly, or vary?)

    Each panel overlays a null (grey): the same statistic computed for simulated
    respondents whose answers are drawn independently from each item's own observed
    distribution. The null is what we'd see with no personal response style. Real
    respondents departing from it (means spread wider; SDs shifted) is the evidence
    that people carry an inherent lean / consistency that is not about the items.
    """
    cols = attitude_columns(df)
    num = df[cols].apply(lambda s: s.map(AGREE_SCALE))   # -2..+2, DK -> NaN

    resp_mean = num.mean(axis=1).dropna()   # baseline lean   (agreeableness)
    resp_std = num.std(axis=1).dropna()      # consistency     (similar vs varied answers)

    # Null: independent random answers. For each item, draw from its OWN observed
    # answers (so each item keeps its real distribution) but break the link to the
    # person -> destroys any per-respondent response style while matching the data.
    #
    # Significance via parametric bootstrap: build n_boot synthetic cohorts, each the
    # same size as the real sample. For every cohort we record a summary statistic, which
    # gives the null sampling distribution of that statistic. Each panel tests a
    # different thing, because each null shares a different feature with the real data:
    #   lean panel : the mean-of-means equals the null by construction, so the only
    #                signal is the SPREAD of per-respondent means (real wider => style).
    #   consistency: the signal is the MEAN of per-respondent SDs -- positive inter-item
    #                correlation makes people answer more alike, so real sits BELOW null.
    rng = np.random.default_rng(0)
    n_boot = 1000
    n_resp = len(resp_mean)                                  # match the real sample size
    item_pools = [num[c].dropna().to_numpy() for c in cols]  # each item's observed answers

    null_means, null_stds = [], []          # pooled per-respondent values -> grey curve
    boot_lean_spread = np.empty(n_boot)     # per-cohort SD-of-means   -> p-value for lean
    boot_consistency = np.empty(n_boot)     # per-cohort mean-of-SDs   -> p-value for consistency
    for b in range(n_boot):
        sims = np.column_stack([rng.choice(pool, n_resp) for pool in item_pools])
        m, s = sims.mean(axis=1), sims.std(axis=1, ddof=1)
        null_means.append(m); null_stds.append(s)
        boot_lean_spread[b] = m.std(ddof=1)
        boot_consistency[b] = s.mean()
    null_mean = np.concatenate(null_means)
    null_std = np.concatenate(null_stds)

    # Empirical one-sided p-values (+1 in num/den so p is never exactly 0).
    # Lean: real spread is larger than null. Consistency: real mean-SD is smaller.
    obs_lean_spread, obs_consistency = resp_mean.std(), resp_std.mean()
    p_lean = (1 + np.sum(boot_lean_spread >= obs_lean_spread)) / (n_boot + 1)
    p_consistency = (1 + np.sum(boot_consistency <= obs_consistency)) / (n_boot + 1)
    fmt_p = lambda p: f"p={p:.3f}" if p > 1 / (n_boot + 1) else f"p<{1/(n_boot+1):.3f}"

    def overlay_null(ax, real, null, color, bins=30):
        """Draw the real histogram, then the null as a comparable grey curve."""
        _, bins, _ = ax.hist(real, bins=bins, color=BAR, edgecolor="white",
                             linewidth=0.4, label="real respondents")
        dens, _ = np.histogram(null, bins=bins, density=True)
        centres = (bins[:-1] + bins[1:]) / 2
        ax.plot(centres, dens * len(real) * (bins[1] - bins[0]), color=color, lw=2.2,
                label="if answers were random & independent")

    BAR, REF, NULLC = "#4C72B0", "#9aa0a6", "#444444"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # --- left: baseline lean = how agreeable is the respondent overall? ----------
    # Fewer bins than the default: means land on a comb spaced 1/11 apart, and a bin
    # width near that spacing produces a picket-fence ripple. Coarser bins span several
    # comb teeth so the ripple averages out.
    ax = axes[0]
    overlay_null(ax, resp_mean, null_mean, NULLC, bins=20)
    ax.axvline(0, color=REF, lw=1.2, ls="--")

    # Horizontal ±1 SD spans near the top: same centre, but the real span is visibly
    # wider than the null -> that width gap IS the response-style signal.
    ymax = ax.get_ylim()[1]
    for centre, spread, color, lbl, yfrac in [
        (resp_mean.mean(), obs_lean_spread, BAR, "real ±1 SD", 0.93),
        (null_mean.mean(), boot_lean_spread.mean(), NULLC, "null ±1 SD", 0.84)]:
        ax.hlines(ymax * yfrac, centre - spread, centre + spread, color=color, lw=2.5)
        ax.plot(centre, ymax * yfrac, "o", color=color, ms=5, label=lbl)
    # ax.set(title=f"Some people lean agree, others disagree — more than chance",
    #        xlabel="← leans disagree        average answer (−2 … +2)        leans agree →",
    #        ylabel="respondents")



    # ax.set(title=f"Lean spread: real SD {obs_lean_spread:.2f} vs null {boot_lean_spread.mean():.2f}  ({fmt_p(p_lean)})",
    #        xlabel="← leans disagree        average answer (−2 … +2)        leans agree →",
    #        ylabel="respondents")
    ax.set(
        xlabel="← leans disagree        average answer (−2 … +2)        leans agree →",
        ylabel="respondents")
    ax.set_title(
        f"Mean responses lean positive but more extreme than chance-level:\nreal SD {obs_lean_spread:.2f} "
        f"vs null {boot_lean_spread.mean():.2f}  (bootstrap {fmt_p(p_lean)})",
        fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)

    # --- right: consistency = similar answers across items, or varied? -----------
    ax = axes[1]
    overlay_null(ax, resp_std, null_std, NULLC, bins=20)
    # Vertical markers at the two means: real sits left of null -> people answer the
    # items more alike than independent chance would produce.
    ax.axvline(obs_consistency, color=BAR, lw=1.8, ls="--", label="real mean")
    ax.axvline(boot_consistency.mean(), color=NULLC, lw=1.8, ls="--", label="null mean")
    ax.set(xlabel="← answers everything similarly      spread across the 11 items (SD)      varied answers →",
           ylabel="respondents")
    ax.set_title((f"Individuals' responses are less varied than chance-level: \nreal mean-SD {obs_consistency:.2f} vs null "
                  f"{boot_consistency.mean():.2f}  (boot-strap {fmt_p(p_consistency)}))"),
                 fontsize=10,
           )

    ax.legend(fontsize=8, loc="upper left", frameon=False)

    share_agree = (resp_mean > 0).mean()
    fig.suptitle(
        "Individuals' response behaviour is biased and lacks diversity",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "02_response_style.png")


# Sensible category orders for the ordinal survey questions, so their bars read
# low->high instead of in arbitrary value-count order. Keyed by clean_data column.
_AGREE_ORDER = sorted(AGREE_SCALE, key=AGREE_SCALE.get) + ["Don't know"]
_FEATURE_ORDERS = {
    encode.INCOME_COL: INCOME_ORDER + ["Prefer not to answer"],
    encode.POLITICAL_INTEREST_COL: list(POLITICAL_INTEREST_SCALE),
    "age": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "general_attitudes_i_consider_myself_working_class": _AGREE_ORDER,
}


def plot_vote_mix_by_feature(df, features, target=TARGET_COL, orders=None,
                             name="03_vote_mix_by_feature.png",
                             title="Vote intention varies most across the features that predict it"):
    """Stacked party-share bars within each category of the given survey features.

    The descriptive companion to the permutation-importance plot ('What matters for
    predicting voting'): for every feature in `features`, one horizontal stacked bar
    per category shows how that category's respondents split across vote intentions
    (crosstab normalised within each row, so bars sum to 100%). This is the whole
    dataset, uncommitted included -- purely descriptive, not the model's prediction.

    features : list of clean_data column names (e.g. region, EU-referendum vote).
    orders   : optional {column: [category order]} to override the default (ordinal
               questions use a low->high order; everything else is most-common-first).
    Mirrors the look of analysis.ipynb's plot_mix, but coloured by party
    (PARTY_COLOURS) rather than by an arbitrary colormap.
    """
    orders = {**_FEATURE_ORDERS, **(orders or {})}
    votes = [p for p in ORDER if p in df[target].unique()]

    # Build the within-category mix for each feature up front so we can size each
    # panel by its number of categories (a 12-band income question needs more room
    # than a 2-way EU vote).
    mixes = []
    for col in features:
        mix = pd.crosstab(df[col], df[target], normalize="index")
        row_order = [r for r in orders.get(col, df[col].value_counts().index) if r in mix.index]
        mix = mix.reindex(index=row_order, columns=votes)
        counts = df[col].value_counts().reindex(row_order)
        mixes.append((col, mix, counts))

    n_rows = [len(mix) for _, mix, _ in mixes]
    fig, axes = plt.subplots(
        len(mixes), 1, figsize=(9.5, 0.45 * sum(n_rows) + 1.2 * len(mixes)),
        gridspec_kw={"height_ratios": n_rows}, squeeze=False)
    axes = axes[:, 0]

    for ax, (col, mix, counts) in zip(axes, mixes):
        left = np.zeros(len(mix))
        for party in votes:
            vals = mix[party].to_numpy()
            ax.barh(range(len(mix)), vals, left=left, color=PARTY_COLOURS[party],
                    edgecolor="white", height=0.8,
                    label=party if ax is axes[0] else None)
            for i, (v, l) in enumerate(zip(vals, left)):
                if v > 0.06:
                    ax.text(l + v / 2, i, f"{v:.0%}", va="center", ha="center",
                            fontsize=7, color="white")
            left += vals
        ax.set_yticks(range(len(mix)),
                      [f"{idx}  (n={int(c)})" for idx, c in zip(mix.index, counts)],
                      fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(pretty_labels(_shorten(col)), fontsize=10, loc="left")
        ax.set_xlabel("share within category" if ax is axes[-1] else "")
        ax.margins(y=0.02)

    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(0.99, 0.97), fontsize=8, title="vote intention")
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 0.83, 0.97))
    save(fig, name)


KDE_MIN_N = 25            # too few points for a trustworthy density below this


def _rgba_by_strength(colour, strength, lo=0.06, hi=1.0, gamma=1.6):
    """Per-point RGBA: alpha scales with opinion strength (0..2).
    gamma>1 steepens the low end so weak/no-opinion points fade harder."""
    a = lo + (hi - lo) * (strength.clip(0, 2) / 2.0) ** gamma
    base = np.array(mcolors.to_rgb(colour))
    return np.column_stack([np.tile(base, (len(strength), 1)), a.values])


def political_values_map(X_all, y_all, opinion_strength,
                         name="04_political_values_map.png"):
    """Place every respondent on the curated (econ, natint) value axes by vote.

    Left: raw points, opacity = opinion strength (faint = neutral / Don't know).
    Right: each group as a KDE density blob with its mean (uncommitted pooled grey).

    opinion_strength : per-respondent mean |score| (0..2) from encoded_data, used
                       only to set point opacity; reindexed to X_all.
    """
    vmap = encode.value_map_scores(X_all)
    vmap["vote"] = y_all
    vmap["strength"] = opinion_strength.reindex(vmap.index).fillna(0)
    unc = vmap[vmap["vote"].isin(UNCOMMITTED)]

    def style(ax, title):
        ax.axhline(0, color="grey", lw=.6)
        ax.axvline(0, color="grey", lw=.6)
        ax.set(xlabel="← internationalist      NATIONALISM–INTERNATIONALISM      nationalist →",
               ylabel="← economic left      ECONOMIC      economic right →", title=title)

    fig, (ax_raw, ax_kde) = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)

    # --- left: raw points, opacity = opinion strength -------------------------
    ax_raw.scatter(unc["natint"], unc["econ"], s=10, c="lightgrey", alpha=0.30,
                   edgecolor="none", label=f"uncommitted ({len(unc)})")
    for party in DECIDED:
        p = vmap[vmap["vote"] == party]
        if p.empty:
            continue
        ax_raw.scatter(p["natint"], p["econ"], s=16,
                       color=_rgba_by_strength(PARTY_COLOURS[party], p["strength"]),
                       edgecolor="none", label=f"{party} ({len(p)})")
    style(ax_raw, "Raw respondents (opacity = opinion strength)")
    ax_raw.legend(loc="upper left", fontsize=7, framealpha=.9)

    # --- right: KDE density per group (uncommitted pooled as grey) --------------
    kde_groups = [(party, PARTY_COLOURS[party]) for party in DECIDED] + [("uncommitted", "#9e9e9e")]
    for party, colour in kde_groups:
        p = unc if party == "uncommitted" else vmap[vmap["vote"] == party]
        if len(p) < KDE_MIN_N:                   # e.g. Plaid Cymru (n=11) shown raw-only
            continue
        sns.kdeplot(x=p["natint"], y=p["econ"], ax=ax_kde, color=colour,
                    fill=True, levels=4, thresh=0.30, alpha=0.28, warn_singular=False)
        sns.kdeplot(x=p["natint"], y=p["econ"], ax=ax_kde, color=colour,
                    levels=[0.30], linewidths=1.5, warn_singular=False)
        ax_kde.scatter(p["natint"].mean(), p["econ"].mean(), s=90, color=colour,
                       edgecolor="white", lw=1.5, zorder=5, label=f"{party} ({len(p)})")
    style(ax_kde, "Group density (KDE core, dot = mean)")
    ax_kde.legend(loc="upper left", fontsize=7, framealpha=.9)

    fig.tight_layout()
    save(fig, name)


IMP_COLOUR = "#4C72B0"


def _importance_dotplot(scores: pd.DataFrame, ref: float, popmean: float,
                        xlabel: str, title: str, name: str, ref_label=None):
    """Ranked horizontal mean +/- 95% CI plot of a per-block score, with significance stars.

    scores : DataFrame with one column per block and one row per replicate (CV fold).
             The 95% CI is the t-based confidence interval of the *mean*, computed from
             the standard deviation across rows: mean +/- t(.975, n-1) * sd / sqrt(n).
    ref    : x of the reference line (0 for the permutation drop, 1/n_classes chance for
             standalone power). popmean : the value the one-sided t-test compares against
             for the stars (usually == ref).
    """
    order = scores.mean().sort_values(ascending=False).index
    d = scores[order]
    m = d.mean()
    n = d.count()
    half = student_t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    pvals = {c: ttest_1samp(d[c].dropna(), popmean, alternative="greater").pvalue
             for c in order}

    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(m.values, y, xerr=half.values, fmt="o", color=IMP_COLOUR,
                ecolor=IMP_COLOUR, elinewidth=1.5, capsize=3, markersize=5, zorder=3)
    ax.set_yticks(y, [pretty_labels(c) for c in order])
    ax.invert_yaxis()                         # most important at the top
    ax.axvline(ref, color="grey", lw=1, ls="--")
    if ref_label:
        ax.text(ref + 0.002, len(order) - 0.4, ref_label,
                va="center", ha="left", fontsize=8, color="grey")
    ax.set(xlabel=xlabel, ylabel="", title=title)

    trans = ax.get_yaxis_transform()
    ax.text(1.04, 1.0, "sig.", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold")
    for i, c in enumerate(order):
        ax.text(1.04, i, stars(pvals[c]), transform=trans, ha="left", va="center",
                fontsize=9, fontweight="bold", clip_on=False)
    fig.subplots_adjust(left=0.26, right=0.88, top=0.92, bottom=0.08)
    save(fig, name)


def permutation_importance_plot(imp: pd.DataFrame, name="06_permutation_importance.png"):
    """Mean +/- 95% CI drop in balanced accuracy when each block is shuffled (ranked).

    imp : output of model.grouped_permutation_importance - one row per (fold, repeat)
          with a '_fold' column. Shuffle repeats are averaged within each fold first, so
          the CI reflects fold-to-fold (generalisation) spread, not per-shuffle noise.
          Stars test (one-sided t) that the per-fold drop is > 0.
    """
    feat_cols = [c for c in imp.columns if c != "_fold"]
    fold_means = imp.groupby("_fold")[feat_cols].mean()
    sns.set_style("whitegrid")
    _importance_dotplot(
        fold_means, ref=0.0, popmean=0.0,
        xlabel="drop in balanced accuracy when shuffled (mean per CV fold +/- 95% CI)",
        title="What matters for predicting voting", name=name)


def standalone_power_plot(uni: pd.DataFrame, n_classes: int,
                          name="07_standalone_power.png"):
    """Mean +/- 95% CI held-out balanced accuracy using each block ALONE (ranked).

    uni : output of model.univariate_score - rows = CV folds, columns = blocks. The
          dashed reference line is the 1/n_classes chance floor; stars test (one-sided t)
          each block against it.
    """
    chance = 1 / n_classes
    sns.set_style("whitegrid")
    _importance_dotplot(
        uni, ref=chance, popmean=chance,
        xlabel="CV balanced accuracy using only this block (mean +/- 95% CI)",
        title="Standalone predictive power of each curated block", name=name,
        ref_label=f"chance ({chance:.0%})")

