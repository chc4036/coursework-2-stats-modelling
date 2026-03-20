import pandas as pd

# 1. Load the weekly dataset
file_path = 'GASDESWf19950101t20260101WEEKLY.csv'
df = pd.read_csv(file_path)

# 2. Prepare the date column
df['observation_date'] = pd.to_datetime(df['observation_date'])
df.set_index('observation_date', inplace=True)

# 3. Resample to Monthly Start frequency ('MS') and calculate the mean
# This will label the average price as the 1st of the month
df_monthly_start = df.resample('MS').mean()

# 4. Clean up and Export
df_monthly_start.reset_index(inplace=True)
df_monthly_start.to_csv('GASDESW_monthly_start.csv', index=False)

print("Transformation complete. New date format:")
print(df_monthly_start.head())
