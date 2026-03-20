import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math

freight_df = pd.read_csv('FRGSHPUSM649NCISf19950101t20260101MONTHLY.csv')
fuel_df = pd.read_csv('GASDESW_monthly_start.csv')

freight_df['observation_date'] = pd.to_datetime(freight_df['observation_date'])
fuel_df['observation_date'] = pd.to_datetime(fuel_df['observation_date'])

merged_df = pd.merge(freight_df, fuel_df, on='observation_date')

merged_df.rename(columns={'FRGSHPUSM649NCIS': 'FreightRate', 'GASDESW': 'FuelPrice'}, inplace=True)

#drop no data rows
merged_df.dropna(subset=['FreightRate', 'FuelPrice'], inplace=True)

X = merged_df['FuelPrice']
Y = merged_df['FreightRate']

#lin regress values
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)


merged_df['Predicted'] = intercept + slope * X
merged_df['Residuals'] = Y - merged_df['Predicted']


beta1 = intercept
beta2 = slope

print(f"y-intercept: {beta1}")
print(f"slope: {beta2}")
print(f"R-squared: {r_value**2}")

#Linear plot
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.5, label='Actual Data')
plt.plot(X, merged_df['Predicted'], color='red', label=f'Fitted Line: Y = {beta1:.4f} + {beta2:.4f}X')
plt.plot([], [], ' ', label=f'$R^2$ ={round(r_value**2,5)}')
plt.title('Freight Rate against Fuel Price')
plt.xlabel('US Diesel Sales Price (GASDESW)')
plt.ylabel('Cass Freight Index: Shipments (FRGSHPUSM649NCIS)')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('linear_model_plot.png')
plt.close()

#sigma hat
sigma_hat = np.sqrt(np.sum(merged_df['Residuals']**2) / (len(X) - 2))

#Leverage for each point (Pii)
leverage = (1/len(X)) + ((X - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
standardized_residuals = merged_df['Residuals'] / (sigma_hat * np.sqrt(1 - leverage))

#Residuals plot
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['Predicted'], standardized_residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Fitted Values')
plt.xlabel('Fitted Values (Predicted Cass Freight Index: Shipments)')
plt.ylabel('Residuals')
plt.grid(True)
plt.savefig('residuals_plot.png')
plt.close()

plt.figure(figsize=(8, 6))
stats.probplot(standardized_residuals, dist="norm", plot=plt)
plt.title('Normal Q-Q Plot of Standardized Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles (Standardized Residuals)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('qq_plot_standardized_residuals.png')
plt.close()
merged_df.to_csv('freight_fuel_regression_results.csv', index=False)

#t-statistic
ssx = np.sum((X - np.mean(X))**2)
rss = np.sum(merged_df['Residuals']**2)
print(beta1*math.sqrt(370*ssx/rss))