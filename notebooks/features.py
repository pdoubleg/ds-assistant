df['age_and_experience'] = df['vehicle_age'] + df['subscription_length']

df.drop(columns=['displacement'], inplace=True)
df['vehicle_load_efficiency'] = df['gross_weight'] / (df['length'] * df['width'])

df.drop(columns=['policy_id'], inplace=True)
df.drop(columns=['is_power_steering'], inplace=True)
df.drop(columns=['airbags'], inplace=True)
df.drop(columns=['segment'], inplace=True)
df['age_to_vehicle_age_ratio'] = df['customer_age'] / (df['vehicle_age'] + 0.1)

df.drop(columns=['is_power_door_locks'], inplace=True)
df.drop(columns=['is_brake_assist'], inplace=True)
