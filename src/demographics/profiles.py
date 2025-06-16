# import pandas as pd
#
# from src.demographics.age import Age
# from src.demographics.country import Country
# from src.demographics.sex import Sex
#
#
# class DemographicProfile:
#
#     def __init__(self, country: type[Country] = None, age: type[Age] = None, sex: type[Sex] = None):
#         self.country = country
#         self.age = age
#         self.sex = sex
#
#     def filter_condition(self, df: pd.DataFrame) -> pd.Series:
#
#         default = df.index.notna()
#         country_condition = self.country.filter_true(df) if self.country is not None else default
#         age_condition = self.age.filter_true(df) if self.age is not None else default
#         sex_condition = self.sex.filter_true(df) if self.sex is not None else default
#
#         return country_condition & age_condition & sex_condition
