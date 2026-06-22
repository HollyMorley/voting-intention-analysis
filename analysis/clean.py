import pandas as pd

def fix_encoding_errors(df: pd.DataFrame) -> pd.DataFrame:
    income_col = "what_is_the_combined_annual_income_of_your_household_prior_to_tax_being_deducted"
    df[income_col] = df[income_col].str.replace("ú", "£", regex=False)
    return df

def drop_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    deduped = df.drop_duplicates()

    # Check if serial_number is now unique
    if deduped["serial_number"].is_unique:
        pass
    else:
        # full-row dedup wasn't enough - there are conflicting records under one ID
        leftover = deduped[deduped.duplicated("serial_number", keep=False)]
        print(
            f"{leftover['serial_number'].nunique()} serials still repeat after dedup - inspect:")
        print(leftover.sort_values("serial_number"))

    return deduped

def remove_items(df: pd.DataFrame, cols_to_remove: list) -> pd.DataFrame:
    df = df.drop(columns=cols_to_remove)
    return df

def fill_missing_as_category(df: pd.DataFrame, columns, label: str = "No answer") -> pd.DataFrame:
    """Fill missing values in the given column(s) with an explicit category label."""
    if isinstance(columns, str):
        columns = [columns]
    df = df.copy()
    df[columns] = df[columns].fillna(label)
    return df


