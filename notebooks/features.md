```python
df['max_torque_value'] = df['max_torque'].apply(lambda x: float(x.split('Nm')[0]) if pd.notnull(x) else np.nan)

df['max_torque_rpm'] = df['max_torque'].apply(lambda x: int(x.split('@')[1].replace('rpm','')) if pd.notnull(x) else np.nan)

df['max_power_value'] = df['max_power'].apply(lambda x: float(x.split('bhp')[0]) if pd.notnull(x) else np.nan)

df['max_power_rpm'] = df['max_power'].apply(lambda x: int(x.split('@')[1].replace('rpm','')) if pd.notnull(x) else np.nan)

df.drop(columns=['max_torque'], inplace=True)
df.drop(columns=['max_power'], inplace=True)
df['power_to_weight_ratio'] = df['max_power_value'] / df['gross_weight'].replace(0, np.nan)

df.drop(columns=['width'], inplace=True)
df['age_subs_ratio'] = df['vehicle_age'] / (df['subscription_length'] + 0.1)  # added 0.1 to avoid division by zero

df.drop(columns=['policy_id'], inplace=True)
df.drop(columns=['is_parking_sensors'], inplace=True)

```