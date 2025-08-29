df['max_power_bhp'] = df['max_power'].str.replace(r'[^\d\.]', '', regex=True).astype(float)

df['max_torque_Nm'] = df['max_torque'].str.replace(r'[^\d\.]', '', regex=True).astype(float)

safety_feats = ['is_brake_assist','is_esc','is_tpms','is_parking_sensors','is_parking_camera','is_front_fog_lights','is_driver_seat_height_adjustable','is_day_night_rear_view_mirror']
df['vehicle_safety_score'] = df[safety_feats].replace({'Yes': 1, 'No': 0}).sum(axis=1)

df['fueltype_by_customerage'] = df['fuel_type'].astype(str) + '_' + df['customer_age'].astype(str)

df['airbags_x_ncap'] = (df['airbags'].fillna(0) * df['ncap_rating'].fillna(0)).astype(int)

df.drop(columns=['length'], inplace=True)
df.drop(columns=['width'], inplace=True)
df.drop(columns=['cylinder'], inplace=True)
df.drop(columns=['gross_weight'], inplace=True)
df.drop(columns=['turning_radius'], inplace=True)
