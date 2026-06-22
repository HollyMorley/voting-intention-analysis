from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = (PROJECT_ROOT / "data" / "raw" /
                 "Stonehaven_tech_test_survey_data 1.csv")

TARGET_COL = \
    "voting_intention_which_party_would_you_vote_for_if_there_was_a_general_election_tomorrow"
DECIDED_PARTIES = ["Conservative", "Labour", "Liberal Democrat", "Other"]

ATTITUDE_PREFIX = "general_attitudes"
AGREE_SCALE = {
    "Disagree strongly": -2,
    "Disagree slightly": -1,
    "Neither agree nor disagree": 0,
    "Agree slightly": 1,
    "Agree strongly": 2,
    # "Don't know" is intentionally absent -> maps to NaN
}

STATEMENTS_PREFIX = "which_of_the_following_statements_apply_to_you_"

# =============================================================================
# Curated feature taxonomy
# -----------------------------------------------------------------------------
# I sort the survey items by what kind of thing they measure:
#
#   VALUES      (general_attitudes_*) -> the two political-values-map axes below.
#   BACKGROUND  (statements_apply_* + working-class self-image) -> "life background":
#               biographical facts, not positions.
#   DEMOGRAPHIC covariates -> predictors / profiling. Stable traits
#               (age, gender, region, income, work).
#   POLITICAL   (political interest + EU referendum vote) -> "political background":
#               political engagement and prior political behaviour.
#   PRIORITIES  (goal_importance) -> A nominal "what do
#               you care about" choice; there is no clear way to collapse it
#               onto a left-right axis, so it stays off the values map.
# =============================================================================

# --- the two value-map axes
# -----------------------------------------------------
# Keys are the attitude stem; the matching feature is f"{ATTITUDE_PREFIX}_{
# stem}__z".
# Sign convention: +1 pushes a respondent toward the axis's right-hand pole.

ECON_AXIS = {  # economic:   left (-)  <->  right (+)
    "there_is_too_much_reliance_on_welfare_and_benefits_in_britain_today": +1,
    "keeping_a_tight_control_over_spending_should_be_the_governments_main_economic_priority": +1,
    "tackling_poverty_and_inequality_should_be_the_governments_top_priority": -1,
    "big_business_takes_advantage_of_ordinary_people": -1,
}

NATINT_AXIS = {
    # nationalism-internationalism:   internationalist (-)  <->  nationalist (+)
    "the_government_should_prioritise_controlling_immigration_over_all_other_policies": +1,
    "government_should_always_put_the_needs_of_british_people_ahead_of_others": +1,
    "britain_is_the_greatest_country_in_the_world": +1,
    "i_worry_that_cuts_in_defence_spending_mean_britain_can_no_longer_defend_itself": +1,
    "britain_is_stronger_when_it_forms_partnerships_with_other_countries": -1,
    "the_uk_should_be_more_outward_looking_in_nature": -1,
}
# Note: 'i_consider_myself_working_class' is the 11th attitude item but I
# think this is more of a
# class indentity statement, not a value -> it lives in BACKGROUND_GROUPS,
# not here. This avoids circularity in exploring values based on background.

# --- life background / experience (biographical facts)
# -------------------------------
BACKGROUND_GROUPS = {
    "socioeconomic": [
        f"{ATTITUDE_PREFIX}_i_consider_myself_working_class__z",
        f"{STATEMENTS_PREFIX}i_own_more_than_one_property",
        f"{STATEMENTS_PREFIX}i_have_done_manual_labour_work_in_the_past",
        f"{STATEMENTS_PREFIX}i_predominantly_earn_my_living_using_my_hands_physical_capabilities",
        f"{STATEMENTS_PREFIX}i_predominantly_earn_a_living_with_my_cognitive_mental_capabilities",
        f"{STATEMENTS_PREFIX}i_had_a_great_education_growing_up",
        f"{STATEMENTS_PREFIX}i_know_how_to_use_microsoft_excel_spreadsheets",
        f"{STATEMENTS_PREFIX}i_have_been_made_redundant_in_my_lifetime",
        f"{STATEMENTS_PREFIX}i_grew_up_in_a_one_parent_household",
    ],
    "cosmopolitan": [
        # behavioural shadow of the nationalism-internationalism axis; expect correlation
        f"{STATEMENTS_PREFIX}i_have_lived_in_london_at_some_point_in_my_life",
        f"{STATEMENTS_PREFIX}i_have_worked_or_lived_abroad_in_my_life",
        f"{STATEMENTS_PREFIX}i_travel_abroad_regularly_at_least_once_every_year_or_so",
        f"{STATEMENTS_PREFIX}i_take_public_transport_regularly",
        f"{STATEMENTS_PREFIX}my_parents_live_nearby",
        # rootedness (inverse of mobility)
    ],
    "vulnerability": [
        f"{STATEMENTS_PREFIX}i_have_a_health_issue_disability_that_affects_the_jobs_i_can_take",
        f"{STATEMENTS_PREFIX}i_care_for_a_child_with_a_serious_health_issue_disability",
    ],
}

# --- demographic covariates
# ---------------------------------------------------
# Pre-one-hot column names (in clean_data); matched to X_all dummies with
# startswith via encode.resolve_columns. Grouped into labelled blocks so each
# survey question stays one unit for importance (its one-hot dummies judged
# together); the friendly keys double as chart labels.
DEMOGRAPHIC_BLOCKS = {
    "actual_age":         ["actual_age"],
    "gender":             ["gender"],
    "region":             ["which_one_of_these_regions_do_you_live_in"],
    "income":             ["income_ordinal", "income_refused"],
    "working_status":     ["which_of_the_following_best_describes_your_current_working_status"],
    "work_organisation":  ["work_organisation_please_tell_us_which_type_of_organisation_you_do_or_did_work_for"],
    "do_you_work_in":     ["do_you_work_in"],
}

# flat, order-preserving list (single source of truth = DEMOGRAPHIC_BLOCKS)
DEMOGRAPHIC_COLS = [c for cols in DEMOGRAPHIC_BLOCKS.values() for c in cols]

# --- political background (engagement + prior vote behaviour)
# ----------------------------
POLITICAL_BLOCKS = {
    "political_interest": ["political_interest_how_much_interest_do_you_generally_have_in_what_is_going_on_in_politics"],
    "eu_referendum":      ["eu_referendum_how_did_you_vote_in_the_eu_referendum_in_june_2016"],
}

# flat, order-preserving list (single source of truth = POLITICAL_BLOCKS)
POLITICAL_COLS = [c for cols in POLITICAL_BLOCKS.values() for c in cols]

# --- engineered response-style / engagement features
# ----------------------------------
# Per-respondent summaries of *how* someone uses the attitude scale. Added back as
# explicit features because within-respondent z-scoring (the __z items + value-map axes)
# deliberately strips this signal out of the per-item values:
#   opinion_mean : baseline lean / acquiescence (do they generally agree or disagree?)
#   opinion_std  : scale usage / extremity (do they use the whole scale or sit mid?)
#   dk_rate      : disengagement (share of attitude items answered 'Don't know')
ENGAGEMENT_BLOCKS = {
    "response_style": ["opinion_mean", "opinion_std"],
    "dk_rate":        ["dk_rate"],
}

# flat, order-preserving list (single source of truth = ENGAGEMENT_BLOCKS)
ENGAGEMENT_COLS = [c for cols in ENGAGEMENT_BLOCKS.values() for c in cols]

# --- priorities (nominal "what do you care about most")
# -------------------------------
# A single nominal choice with no natural left-right ordering, so it stays off the
# values map, but it is a real survey block -> kept in the curated model as one-hot dummies.
PRIORITY_BLOCKS = {
    "goal_importance": ["goal_importance_most_important"],
}

# flat, order-preserving list (single source of truth = PRIORITY_BLOCKS)
PRIORITY_COLS = [c for cols in PRIORITY_BLOCKS.values() for c in cols]


