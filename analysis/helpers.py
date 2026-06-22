import pandas as pd

from config import ATTITUDE_PREFIX

def attitude_columns(df: pd.DataFrame) -> list[str]:
    """The attitudinal (agree/disagree) survey items."""
    return [c for c in df.columns if c.startswith(ATTITUDE_PREFIX)]

def starts_with(prefix: str):
    """Factory: build a function that selects column names beginning with `prefix`."""
    return lambda columns: [c for c in columns if c.startswith(prefix)]

COL_MAPPING_FNS = {
    "age":                starts_with("age"),
    "gender":             starts_with("gender"),
    "working_status":     starts_with("which_of_the_following_best_describes_your_current_working_status"),
    "region":             starts_with("which_one_of_these_regions_do_you_live_in"),
    "income":             starts_with("income"),          # encode_income renames -> income_ordinal / income_refused
    "work_organisation":  starts_with("work_organisation"),
    "do_you_work_in":     starts_with("do_you_work_in"),
    "political_interest": starts_with("political_interest"),
    "eu_referendum":      starts_with("eu_referendum"),
    "statements_apply":   starts_with("which_of_the_following_statements_apply_to_you"),
    "general_attitudes":  starts_with("general_attitudes"),
    "goal_importance":    starts_with("goal_importance"),
}

from config import ATTITUDE_PREFIX, TARGET_COL      # add TARGET_COL to the import

INCOME_COL = "what_is_the_combined_annual_income_of_your_household_prior_to_tax_being_deducted"

# friendly labels for the long multi-category questions (each stays one block)
FRIENDLY = {
    "gender": "gender",
    "which_one_of_these_regions_do_you_live_in": "region",
    "which_of_the_following_best_describes_your_current_working_status": "working_status",
    "work_organisation_please_tell_us_which_type_of_organisation_you_do_or_did_work_for": "work_organisation",
    "do_you_work_in": "do_you_work_in",
    "eu_referendum_how_did_you_vote_in_the_eu_referendum_in_june_2016": "eu_referendum",
    "political_interest_how_much_interest_do_you_generally_have_in_what_is_going_on_in_politics": "political_interest",
    "goal_importance_most_important": "goal_importance",
    INCOME_COL: "income",
}

def _shorten(name: str) -> str:
    """Group label: friendly name for block questions, else strip the family prefix."""
    if name in FRIENDLY:
        return FRIENDLY[name]
    return (name.replace("general_attitudes_", "")
                .replace("which_of_the_following_statements_apply_to_you_", ""))

def build_column_groups(encoded_columns, original_columns, skip=("serial_number",)):
    """One importance group per survey *question*.

    Each original survey column maps to the encoded column(s) it produced, so a
    multi-category question (region, i_own_more_than_one_property) stays together
    as one block, while every attitude item and every 'statements apply' item is
    its own group — they are unrelated questions and must not be lumped.
    """
    skip = set(skip) | {TARGET_COL}
    groups, claimed = {}, set()

    for orig in original_columns:
        if orig in skip:
            continue
        prefix = "income" if orig == INCOME_COL else orig   # encode_income renames this
        cols = [c for c in encoded_columns if c.startswith(prefix)]
        if cols:
            groups[_shorten(orig)] = cols
            claimed.update(cols)

    for c in encoded_columns:            # engineered extras (dk_rate, opinion_*) stand alone
        if c not in claimed:
            groups[c] = [c]
    return groups

# --- readable chart labels (keyed by the group names build_column_groups produces) ---
LABELS = {
    # attitudes
    "tackling_poverty_and_inequality_should_be_the_governments_top_priority": "tackle poverty",
    "the_government_should_prioritise_controlling_immigration_over_all_other_policies": "control immigration",
    "there_is_too_much_reliance_on_welfare_and_benefits_in_britain_today": "welfare reliance",
    "britain_is_stronger_when_it_forms_partnerships_with_other_countries": "intl. partnerships",
    "keeping_a_tight_control_over_spending_should_be_the_governments_main_economic_priority": "tight spending",
    "government_should_always_put_the_needs_of_british_people_ahead_of_others": "British people first",
    "big_business_takes_advantage_of_ordinary_people": "big business exploits",
    "britain_is_the_greatest_country_in_the_world": "Britain is greatest",
    "i_consider_myself_working_class": "working-class identity",
    "i_worry_that_cuts_in_defence_spending_mean_britain_can_no_longer_defend_itself": "defence worries",
    "the_uk_should_be_more_outward_looking_in_nature": "outward-looking UK",
    # life background
    "i_have_a_health_issue_disability_that_affects_the_jobs_i_can_take": "health affects work",
    "i_predominantly_earn_a_living_with_my_cognitive_mental_capabilities": "works with mind",
    "i_predominantly_earn_my_living_using_my_hands_physical_capabilities": "works with hands",
    "i_have_done_manual_labour_work_in_the_past": "past manual labour",
    "i_have_been_made_redundant_in_my_lifetime": "been redundant",
    "i_care_for_a_child_with_a_serious_health_issue_disability": "cares for disabled child",
    "i_grew_up_in_a_one_parent_household": "one-parent upbringing",
    "i_had_a_great_education_growing_up": "great education",
    "i_know_how_to_use_microsoft_excel_spreadsheets": "knows Excel",
    "i_have_lived_in_london_at_some_point_in_my_life": "lived in London",
    "i_have_worked_or_lived_abroad_in_my_life": "lived abroad",
    "i_take_public_transport_regularly": "public transport",
    "i_travel_abroad_regularly_at_least_once_every_year_or_so": "travels abroad",
    "i_own_more_than_one_property": "owns 2nd property",
    "my_parents_live_nearby": "parents nearby",
    "none_of_these": "none of these",
    # curated blocks (value axes + life-background sub-domains)
    "econ": "economic axis", "natint": "nationalism–internationalism axis",
    "socioeconomic": "socioeconomic background","cosmopolitan": "cosmopolitan background",
    "vulnerability": "vulnerability",
    # blocks / extras
    "eu_referendum": "EU referendum vote", "goal_importance": "top global goal",
    "working_status": "working status", "work_organisation": "work organisation",
    "do_you_work_in": "public-sector field", "political_interest": "political interest",
    "actual_age": "age", "dk_rate": "don't-know rate",
    "opinion_mean": "opinion lean", "opinion_std": "opinion spread",
}

def pretty_labels(name):
    return LABELS.get(name, name.replace("_", " "))

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."