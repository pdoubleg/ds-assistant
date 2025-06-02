```python
df['max_torque_value'] = df['max_torque'].apply(lambda x: float(x.split('Nm@')[0]) if isinstance(x, str) else np.nan)

df['max_power_value'] = df['max_power'].apply(lambda x: float(x.split('bhp@')[0]) if isinstance(x, str) else np.nan)

df['age_to_vehicle_ratio'] = df['customer_age'] / (1 + df['vehicle_age'])

safety_cols = ['is_esc','is_adjustable_steering','is_tpms','is_parking_sensors','is_parking_camera','is_front_fog_lights','is_rear_window_wiper','is_rear_window_washer','is_rear_window_defogger','is_brake_assist','is_power_door_locks','is_central_locking','is_power_steering','is_driver_seat_height_adjustable','is_day_night_rear_view_mirror','is_ecw','is_speed_alert']
for col in safety_cols:
    df[col+'_bin'] = df[col].map({'Yes': 1, 'No': 0})
df['safety_feature_count'] = df[[col+'_bin' for col in safety_cols]].sum(axis=1)

df['torque_power_ratio'] = df['max_torque_value'] / df['max_power_value']

df.drop(columns=['length'], inplace=True)
df.drop(columns=['width'], inplace=True)
df.drop(columns=['displacement'], inplace=True)
df.drop(columns=['turning_radius'], inplace=True)
df.drop(columns=['gross_weight'], inplace=True)
df['customer_vehicle_age_interaction'] = df['customer_age'] * df['vehicle_age']

df['power_safety_index'] = (df['max_torque_value'] / df['max_torque_value'].max()) * (df['ncap_rating'] / (df['ncap_rating'].max() + 1e-6))

segment_mapping = {'A': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5, 'Utility': 6}
df['segment_ordinal'] = df['segment'].map(segment_mapping).fillna(0).astype(int)

df['subscription_safety_interaction'] = df['subscription_length'] * df['safety_feature_count']

df['old_vehicle_low_ncap'] = ((df['vehicle_age'] > 5) & (df['ncap_rating'] <= 1)).astype(int)

df.drop(columns=['max_power_value'], inplace=True)
df['vehicle_risk_age_index'] = df['vehicle_age'] * (5 - df['ncap_rating'])

df['log_region_density'] = np.log1p(df['region_density'])

df['safety_feature_per_age'] = df['safety_feature_count'] / (df['vehicle_age'] + 1)  # +1 to avoid division by zero

df['high_torque_power_ratio'] = (df['torque_power_ratio'] > 1.8).astype(int)

df['sub_length_airbag_index'] = df['subscription_length'] * (df['airbags'] / df['airbags'].max())

df.drop(columns=['segment_ordinal'], inplace=True)
df.drop(columns=['safety_feature_count'], inplace=True)

```