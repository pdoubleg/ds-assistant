# Convert max_power to numeric by extracting numerical value
# Note: Using .split() to handle the conversion manually
df['max_power_numeric'] = df['max_power'].apply(lambda x: float(x.split('bhp')[0]))
# Calculate the power to weight ratio
df['power_to_weight_ratio'] = df['max_power_numeric'] / df['gross_weight']

df.drop(columns=['policy_id'], inplace=True)
df.drop(columns=['is_parking_sensors'], inplace=True)
df.drop(columns=['is_speed_alert'], inplace=True)
