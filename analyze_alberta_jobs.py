import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv('alberta_job_vacancies.csv')

# Display basic info
print("Data shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Filter for relevant rows: Total, all occupations and Average offered hourly wage
# Assuming the dataset is already for Alberta only (based on file name)
mask = (df['National Occupational Classification'] == 'Total, all occupations') & \
       (df['Job vacancy characteristics'] == 'Duration of job vacancy, all durations') & \
       (df['Statistics'] == 'Average offered hourly wage')
df_filtered = df[mask].copy()

print("\nFiltered rows shape:", df_filtered.shape)
print(df_filtered[['REF_DATE', 'VALUE']].head())

# Convert REF_DATE to datetime
df_filtered['REF_DATE'] = pd.to_datetime(df_filtered['REF_DATE'])
df_filtered = df_filtered.sort_values('REF_DATE')

# Plot average offered hourly wage over time
plt.figure(figsize=(10, 6))
plt.plot(df_filtered['REF_DATE'], df_filtered['VALUE'], marker='o')
plt.title('Average Offered Hourly Wage for All Occupations in Alberta')
plt.xlabel('Date')
plt.ylabel('Average Offered Hourly Wage (Dollars)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('alberta_avg_wage_trend.png')
print("\nPlot saved as alberta_avg_wage_trend.png")

# Compute some statistics
print("\nSummary statistics for average offered hourly wage:")
print(df_filtered['VALUE'].describe())

# Year-over-year change (if we have enough data)
df_filtered['YoY_change'] = df_filtered['VALUE'].pct_change(periods=4) * 100  # quarterly data, 4 quarters = 1 year
print("\nLatest YoY change (%):")
print(df_filtered[['REF_DATE', 'VALUE', 'YoY_change']].tail())

# Save processed data
df_filtered.to_csv('alberta_job_vacancies_processed.csv', index=False)
print("\nProcessed data saved to alberta_job_vacancies_processed.csv")