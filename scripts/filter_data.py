import os

import pandas as pd

from data.filtering import filter_by_subgroups
from demographics.base import BaseSubGroup
from demographics.config import ALL_COUNTRIES, ALL_SEXES, ALL_AGES


def main(directory: str, subgroups: list[type[BaseSubGroup]]):
    file_name = "../data_files/WV7/WVS_Cross-National_Wave_7_csv_v6_0.csv"
    df_all = pd.read_csv(os.path.join(directory, file_name))
    subgroup_dfs = {}
    for subgroup in subgroups:
        subgroup_dfs[subgroup.NAME] = filter_by_subgroups(df_all, [subgroup])

    df_directory = os.path.join(directory, "dataframes")
    if not os.path.exists(df_directory):
        os.makedirs(df_directory)

    for name, df in subgroup_dfs.items():
        df.to_csv(os.path.join(df_directory, f"{name}.csv"))


if __name__ == "__main__":
    wd = os.getcwd()  # change as needed
    os.chdir(wd)
    main("../data_files/WV7", ALL_COUNTRIES + ALL_SEXES + ALL_AGES)
