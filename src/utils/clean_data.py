import pandas as pd


def clean_data(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.drop("litres_paid_in_passbook", axis=1)
    train_df.loc[(train_df.village_cluster == "Cluster A"), "feed_kg"] = train_df.loc[
        (train_df.village_cluster == "Cluster A"), "feed_kg"
    ].map(lambda x: x / 1e03)
    train_df = train_df.drop_duplicates()
    return train_df
