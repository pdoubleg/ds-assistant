from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Numeric features (salary columns) - very high missingness, here we impute with a constant
numeric_features = ["salary_year_avg", "salary_hour_avg"]

# Binary features
binary_features = [
    "job_work_from_home",
    "job_no_degree_mention",
    "job_health_insurance",
]

# Categorical features
categorical_features = [
    "job_title_short",
    "job_title",
    "job_location",
    "job_via",
    "job_schedule_type",
    "search_location",
    "job_country",
    "salary_rate",
    "company_name",
]

# Date feature
date_feature = ["job_posted_date"]


# Custom transformer to parse date column into numeric features
class DateParser(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dates = pd.to_datetime(X[self.date_column], errors="coerce")
        df_dates = pd.DataFrame()
        df_dates["year"] = dates.dt.year.fillna(0).astype(int)
        df_dates["month"] = dates.dt.month.fillna(0).astype(int)
        df_dates["day"] = dates.dt.day.fillna(0).astype(int)
        df_dates["dayofweek"] = dates.dt.dayofweek.fillna(0).astype(int)
        return df_dates.values


# Custom transformer to extract count of skills (assuming skills are list-like strings)
class SkillCountExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, skill_column):
        self.skill_column = skill_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def count_skills(x):
            if pd.isnull(x):
                return 0
            try:
                skills_list = eval(x)
                if isinstance(skills_list, list):
                    return len(skills_list)
                else:
                    return 0
            except:
                return 0

        skill_counts = X[self.skill_column].apply(count_skills)
        return skill_counts.values.reshape(-1, 1)


# Define preprocessing pipelines for different feature types
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("scaler", StandardScaler()),
    ]
)

binary_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

date_transformer = Pipeline(
    steps=[("date_parser", DateParser(date_column="job_posted_date"))]
)

skills_transformer = Pipeline(
    steps=[("skill_counts", SkillCountExtractor(skill_column="job_skills"))]
)

# Combine all transformers in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("bin", binary_transformer, binary_features),
        ("cat", categorical_transformer, categorical_features),
        ("date", date_transformer, date_feature),
        ("skills", skills_transformer, ["job_skills"]),
    ],
    remainder="drop",  # Drop other columns not specified
)

# Final pipeline
cleaning_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Usage example:
# cleaned_data = cleaning_pipeline.fit_transform(df)
