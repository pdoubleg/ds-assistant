df['max_torque_nm'] = df['max_torque'].apply(lambda x: float(x.split('Nm@')[0]) if pd.notnull(x) else np.nan)

df['max_power_bhp'] = df['max_power'].apply(lambda x: float(x.split('bhp@')[0]) if pd.notnull(x) else np.nan)

df['vehicle_age_subscription_length_interaction'] = df['vehicle_age'] * df['subscription_length']

safety_features = ['is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 'is_parking_camera', 'is_front_fog_lights', 'is_rear_window_wiper', 'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks', 'is_central_locking', 'is_power_steering', 'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror', 'is_ecw', 'is_speed_alert']
for col in safety_features:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['safety_equipment_count'] = df[safety_features].sum(axis=1)

df['length_to_width_ratio'] = df['length'] / df['width']

bins = [30, 40, 50, 60, 70, 80]
labels = ['30-39', '40-49', '50-59', '60-69', '70-79']
df['customer_age_group'] = pd.cut(df['customer_age'], bins=bins, labels=labels, right=False)

df.drop(columns=['policy_id'], inplace=True)
df['subscription_to_vehicle_age_ratio'] = df['subscription_length'] / (df['vehicle_age'] + 1)  # +1 to avoid division by zero

df['torque_to_displacement_ratio'] = df['max_torque_nm'] / df['displacement']

df['power_to_displacement_ratio'] = df['max_power_bhp'] / df['displacement']

bins = [34, 39, 44, 49, 54, 59, 75]
labels = ['35-39', '40-44', '45-49', '50-54', '55-59', '60+']
df['customer_age_fine_group'] = pd.cut(df['customer_age'], bins=bins, labels=labels, right=True, include_lowest=True)

df['log_region_density'] = np.log1p(df['region_density'])

df['has_advanced_safety_features'] = ((df['is_esc'] + df['is_tpms'] + df['is_parking_camera'] + df['is_brake_assist']) > 0).astype(int)

df.drop(columns=['max_power_bhp'], inplace=True)
df.drop(columns=['length_to_width_ratio'], inplace=True)
df['age_gap'] = df['customer_age'] - df['vehicle_age']

df['engine_power_density'] = df['max_torque_nm'] / df['displacement']

df['safety_airbags_interaction'] = df['airbags'] * df['safety_equipment_count']

df['subscription_length_bucket'] = pd.cut(df['subscription_length'], bins=[0,2,5,10,15], labels=['Short','Medium','Long','Very Long'], include_lowest=True)

segment_freq = df['segment'].value_counts(normalize=True)
df['freq_segment'] = df['segment'].map(segment_freq)

df.drop(columns=['vehicle_age_subscription_length_interaction'], inplace=True)
df.drop(columns=['torque_to_displacement_ratio'], inplace=True)
df.drop(columns=['power_to_displacement_ratio'], inplace=True)
df.drop(columns=['safety_equipment_count'], inplace=True)
df.drop(columns=['customer_age_fine_group'], inplace=True)
