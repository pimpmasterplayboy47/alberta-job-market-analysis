import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the CSV
df = pd.read_csv('alberta_job_vacancies.csv')
print("Data loaded:", df.shape)

# Filter for Average offered hourly wage (only statistic available)
df = df[df['Statistics'] == 'Average offered hourly wage'].copy()
print("After filtering for Average offered hourly wage:", df.shape)

# Convert REF_DATE to datetime
df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])

# We'll analyze by National Occupational Classification (major groups)
# Let's get the unique occupations (excluding 'Total, all occupations' for detailed view)
occupations = df['National Occupational Classification'].unique()
print("Unique occupations:", occupations)

# Create a pivot table: average wage by occupation and date
# We'll use VALUE as the average offered hourly wage
pivot = df.pivot_table(index='REF_DATE', columns='National Occupational Classification', values='VALUE', aggfunc='mean')
print("\nPivot table shape:", pivot.shape)

# Plot each occupation's wage over time
plt.figure(figsize=(14, 8))
for col in pivot.columns:
    if col == 'Total, all occupations':
        # Plot total with a thicker line
        plt.plot(pivot.index, pivot[col], label=col, linewidth=3, color='black')
    else:
        plt.plot(pivot.index, pivot[col], label=col, linewidth=2)
plt.title('Average Offered Hourly Wage by Occupation Group in Alberta (2015-2023)')
plt.xlabel('Date')
plt.ylabel('Average Offered Hourly Wage (Dollars)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('alberta_wage_by_occupation.png', dpi=300)
print("\nSaved occupation wage plot to alberta_wage_by_occupation.png")

# Calculate overall statistics for each occupation
print("\n=== Summary Statistics by Occupation (2015-2023) ===")
for col in pivot.columns:
    series = pivot[col].dropna()
    if len(series) > 0:
        print(f"{col}:")
        print(f"  Mean: ${series.mean():.2f}")
        print(f"  Median: ${series.median():.2f}")
        print(f"  Min: ${series.min():.2f}")
        print(f"  Max: ${series.max():.2f}")
        print(f"  Latest (2023-07): ${series.iloc[-1]:.2f}")
        print()

# Compute year-over-year change for each occupation (latest available)
print("=== Year-over-Year Change (Latest Quarter vs Same Quarter Previous Year) ===")
# Assuming quarterly data, we need same quarter previous year: shift by 4 quarters
for col in pivot.columns:
    series = pivot[col].dropna()
    if len(series) >= 5:  # need at least 5 points to have YoY
        latest = series.iloc[-1]
        year_ago = series.iloc[-5] if len(series) >=5 else None
        if year_ago is not None and year_ago != 0:
            yoy_pct = ((latest - year_ago) / year_ago) * 100
            print(f"{col}: {yoy_pct:+.2f}% (${latest:.2f} vs ${year_ago:.2f} a year ago)")
        else:
            print(f"{col}: Insufficient data for YoY")
    else:
        print(f"{col}: Insufficient data")

# Save processed pivot table
pivot.to_csv('alberta_wage_by_occupation_pivot.csv')
print("\nSaved pivot table to alberta_wage_by_occupation_pivot.csv")

plt.show()
