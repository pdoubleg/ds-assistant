# === FEATURE_ENGINEERING_CODE ===
df["max_torque_nm"] = df["max_torque"].apply(
    lambda x: float(
        str(x).split("Nm@")[0].replace(" ", "").replace("nm", "").replace("NM", "")
    )
    if pd.notnull(x) and "Nm@" in str(x)
    else np.nan
)
df["max_torque_rpm"] = df["max_torque"].apply(
    lambda x: float(
        str(x).split("@")[1].replace("rpm", "").replace("RPM", "").replace(" ", "")
    )
    if pd.notnull(x) and "@" in str(x)
    else np.nan
)

df["max_power_bhp"] = df["max_power"].apply(
    lambda x: float(
        str(x).split("bhp@")[0].replace(" ", "").replace("BHP", "").replace("bhp", "")
    )
    if pd.notnull(x) and "bhp@" in str(x)
    else np.nan
)
df["max_power_rpm"] = df["max_power"].apply(
    lambda x: float(
        str(x).split("@")[1].replace("rpm", "").replace("RPM", "").replace(" ", "")
    )
    if pd.notnull(x) and "@" in str(x)
    else np.nan
)

df["vehicle_usage_ratio"] = df["subscription_length"] / (df["vehicle_age"] + 0.1)

safety_cols = [
    "is_esc",
    "is_tpms",
    "is_parking_sensors",
    "is_parking_camera",
    "is_front_fog_lights",
    "is_rear_window_wiper",
    "is_rear_window_washer",
    "is_rear_window_defogger",
    "is_brake_assist",
    "is_power_door_locks",
    "is_central_locking",
    "is_driver_seat_height_adjustable",
    "is_day_night_rear_view_mirror",
    "is_ecw",
]
df["safety_feature_score"] = df[safety_cols].apply(
    lambda r: np.sum([1 if str(v).strip().lower() == "yes" else 0 for v in r]), axis=1
)

df["power_to_weight_ratio"] = df["max_power_bhp"] / df["gross_weight"].replace(
    0, np.nan
)

df.drop(columns=["policy_id"], inplace=True)
df.drop(columns=["is_power_steering"], inplace=True)
df.drop(columns=["is_speed_alert"], inplace=True)
df.drop(columns=["max_torque"], inplace=True)
df.drop(columns=["max_power"], inplace=True)
df.drop(columns=["length"], inplace=True)
df.drop(columns=["turning_radius"], inplace=True)

df["exposure_minus_vehicle_age"] = df["subscription_length"] - df["vehicle_age"]

df["customer_vehicle_age_gap"] = df["customer_age"] - df["vehicle_age"]

df["log_region_density"] = np.log1p(df["region_density"])

df["ncap_per_safety_feature"] = df["ncap_rating"] / (df["safety_feature_score"] + 1)

df["torque_band_ratio"] = df["max_torque_nm"] / df["max_torque_rpm"].replace(0, np.nan)

df.drop(columns=["safety_feature_score"], inplace=True)
df.drop(columns=["power_to_weight_ratio"], inplace=True)

# === FEATURE_ENCODING_CODE ===
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numeric_features = [
    "subscription_length",
    "vehicle_age",
    "customer_age",
    "region_density",
    "airbags",
    "displacement",
    "cylinder",
    "width",
    "gross_weight",
    "ncap_rating",
    "max_torque_nm",
    "max_torque_rpm",
    "max_power_bhp",
    "max_power_rpm",
    "vehicle_usage_ratio",
    "exposure_minus_vehicle_age",
    "customer_vehicle_age_gap",
    "log_region_density",
    "ncap_per_safety_feature",
    "torque_band_ratio",
]

categorical_features = [
    "region_code",
    "segment",
    "model",
    "fuel_type",
    "engine_type",
    "is_esc",
    "is_adjustable_steering",
    "is_tpms",
    "is_parking_sensors",
    "is_parking_camera",
    "rear_brakes_type",
    "transmission_type",
    "steering_type",
    "is_front_fog_lights",
    "is_rear_window_wiper",
    "is_rear_window_washer",
    "is_rear_window_defogger",
    "is_brake_assist",
    "is_power_door_locks",
    "is_central_locking",
    "is_driver_seat_height_adjustable",
    "is_day_night_rear_view_mirror",
    "is_ecw",
]

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=0.01,
                sparse_output=True,
            ),
        ),
    ]
)

encoder = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)
