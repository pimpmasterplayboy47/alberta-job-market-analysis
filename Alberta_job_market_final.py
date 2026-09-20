import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


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



sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_and_prepare_data():
    """Load and prepare the Alberta job vacancies data"""
    df = pd.read_csv('alberta_job_vacancies.csv')


    # Filter for Average offered hourly wage (only statistic available)
    df = df[df['Statistics'] == 'Average offered hourly wage'].copy()


    # Convert REF_DATE to datetime
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])


    return df


def calculate_job_quality_metrics(df):
    """Calculate multiple metrics to assess job quality for each occupation"""


    # Get unique occupations (excluding 'Total, all occupations' and 'Unclassified' for cleaner analysis)
    occupations = df['National Occupational Classification'].unique()
    occupations = [occ for occ in occupations if occ not in ['Total, all occupations', 'Unclassified occupations']]


    metrics_list = []


    for occupation in occupations:
        # Filter data for this occupation
        occ_data = df[df['National Occupational Classification'] == occupation].copy()
        occ_data = occ_data.sort_values('REF_DATE')


        if len(occ_data) < 4:  # Need at least some data points
            continue


        # Extract wage series
        wages = occ_data['VALUE'].values
        dates = occ_data['REF_DATE'].values


        # Calculate metrics


        # 1. Current wage level (latest)
        current_wage = wages[-1]


        # 2. Wage growth (year-over-year % change)
        if len(wages) >= 5:  # Need at least 5 quarters for YoY (4 quarters back + current)
            wage_yoy = ((wages[-1] - wages[-5]) / wages[-5]) * 100
        else:
            wage_yoy = np.nan


        # 3. Wage stability (inverse of coefficient of variation)
        # Lower CV = more stable = better
        wage_mean = np.mean(wages)
        wage_std = np.std(wages)
        if wage_mean > 0:
            cv = wage_std / wage_mean
            stability_score = 1 / (1 + cv)  # Transform to 0-1 range where higher is better
        else:
            stability_score = np.nan


        # 4. Growth consistency (R-squared of linear trend)
        # Higher R-squared = more consistent trend = better
        if len(wages) >= 3:
            x = np.arange(len(wages))
            # Simple linear regression
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, wages, rcond=None)[0]
            predicted = slope * x + intercept
            ss_res = np.sum((wages - predicted) ** 2)
            ss_tot = np.sum((wages - np.mean(wages)) ** 2)
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                # Ensure R-squared is between 0 and 1
                r_squared = max(0, min(1, r_squared))
            else:
                r_squared = 0
        else:
            r_squared = np.nan


        # 5. Relative position (how current wage compares to historical median)
        historical_median = np.median(wages)
        if historical_median > 0:
            relative_position = (current_wage - historical_median) / historical_median
        else:
            relative_position = np.nan


        # 6. Recent momentum (last 2 quarters change)
        if len(wages) >= 3:
            recent_change = ((wages[-1] - wages[-3]) / wages[-3]) * 100
        else:
            recent_change = np.nan


        metrics_list.append({
            'Occupation': occupation,
            'Current_Wage': current_wage,
            'Wage_Growth_YoY': wage_yoy,
            'Stability_Score': stability_score,
            'Trend_Consistency': r_squared,
            'Relative_Position': relative_position,
            'Recent_Change': recent_change,
            'Data_Points': len(wages),
            'Start_Date': dates[0],
            'End_Date': dates[-1]
        })


    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df


def normalize_metrics(metrics_df):
    """Normalize metrics to 0-1 scale for scoring"""
    df_norm = metrics_df.copy()


    # Define which metrics should be maximized (higher is better)
    maximize_metrics = ['Current_Wage', 'Wage_Growth_YoY', 'Stability_Score',
                       'Trend_Consistency', 'Relative_Position', 'Recent_Change']


    for metric in maximize_metrics:
        if metric in df_norm.columns:
            # Handle NaN values
            valid_vals = df_norm[metric].dropna()
            if len(valid_vals) > 0:
                min_val = valid_vals.min()
                max_val = valid_vals.max()
                if max_val > min_val:
                    df_norm[f'{metric}_Normalized'] = (df_norm[metric] - min_val) / (max_val - min_val)
                else:
                    df_norm[f'{metric}_Normalized'] = 0.5  # All same value
            else:
                df_norm[f'{metric}_Normalized'] = np.nan
        else:
            df_norm[f'{metric}_Normalized'] = np.nan


    return df_norm


def calculate_composite_score(df_norm, weights=None):
    """Calculate composite score based on normalized metrics"""
    if weights is None:
        # Equal weights by default
        weights = {
            'Current_Wage': 0.25,
            'Wage_Growth_YoY': 0.20,
            'Stability_Score': 0.20,
            'Trend_Consistency': 0.15,
            'Relative_Position': 0.10,
            'Recent_Change': 0.10
        }


    # Initialize score column
    df_norm['Composite_Score'] = np.nan


    # Calculate composite score as weighted average of available normalized metrics
    norm_cols = []
    for metric in weights.keys():
        norm_col = f'{metric}_Normalized'
        if norm_col in df_norm.columns:
            norm_cols.append(norm_col)


    if norm_cols:
        # For each row, calculate weighted average of available normalized metrics
        for idx in df_norm.index:
            available_norms = []
            available_weights = []
            for metric, weight in weights.items():
                norm_col = f'{metric}_Normalized'
                if norm_col in df_norm.columns and not pd.isna(df_norm.loc[idx, norm_col]):
                    available_norms.append(df_norm.loc[idx, norm_col])
                    available_weights.append(weight)


            if len(available_norms) > 0:
                # Weighted average
                weighted_sum = np.sum(np.array(available_norms) * np.array(available_weights))
                total_weight = np.sum(available_weights)
                df_norm.loc[idx, 'Composite_Score'] = weighted_sum / total_weight
            # else remains NaN


    return df_norm


def visualize_results(metrics_df_with_scores):
    """Create visualizations for the job quality analysis"""


    # 1. Scatter plot: Current Wage vs Wage Growth
    plt.figure(figsize=(14, 10))


    plt.subplot(2, 2, 1)
    valid_data = metrics_df_with_scores.dropna(subset=['Current_Wage', 'Wage_Growth_YoY'])
    if len(valid_data) > 0:
        scatter = plt.scatter(valid_data['Current_Wage'], valid_data['Wage_Growth_YoY'],
                             s=valid_data['Composite_Score']*200 + 50, alpha=0.7,
                             c=valid_data['Composite_Score'], cmap='viridis')
        plt.colorbar(scatter, label='Composite Score')
        plt.xlabel('Current Wage ($/hour)')
        plt.ylabel('Wage Growth YoY (%)')
        plt.title('Job Quality: Wage Level vs Growth\n(Bubble size = Composite Score)')
        plt.grid(True, alpha=0.3)


    # 2. Bar chart: Top 10 jobs by composite score
    plt.subplot(2, 2, 2)
    top_10 = metrics_df_with_scores.nlargest(10, 'Composite_Score')
    # Shorten occupation names for display
    top_10['Short_Name'] = top_10['Occupation'].str.replace(' occupations', '').str.replace('Natural and applied sciences and related', 'Sciences').str.replace('Business, finance and administration', 'Business').str.replace('Trades, transport and equipment operators and related', 'Trades').str.replace('Sales and service', 'Sales/Services').str.replace('Occupations in ', '')
    if len(top_10) > 0:
        plt.barh(range(len(top_10)), top_10['Composite_Score'])
        plt.yticks(range(len(top_10)), top_10['Short_Name'])
        plt.xlabel('Composite Score')
        plt.title('Top 10 Best Jobs in Alberta\n(Based on Multiple Factors)')
        plt.gca().invert_yaxis()


    # 3. Histogram of composite scores
    plt.subplot(2, 2, 3)
    valid_scores = metrics_df_with_scores['Composite_Score'].dropna()
    if len(valid_scores) > 0:
        plt.hist(valid_scores, bins=15, edgecolor='black', alpha=0.7)
        plt.xlabel('Composite Score')
        plt.ylabel('Number of Occupations')
        plt.title('Distribution of Job Quality Scores')
        plt.grid(True, alpha=0.3)


    # 4. Wage trends for top 3 jobs
    plt.subplot(2, 2, 4)
    top_3 = metrics_df_with_scores.nlargest(3, 'Composite_Score')
    df_raw = load_and_prepare_data()


    for _, row in top_3.iterrows():
        occupation = row['Occupation']
        occ_data = df_raw[df_raw['National Occupational Classification'] == occupation].sort_values('REF_DATE')
        if len(occ_data) > 0:
            plt.plot(occ_data['REF_DATE'], occ_data['VALUE'],
                    label=occupation.replace(' occupations', ''), linewidth=2, marker='o')


    plt.xlabel('Date')
    plt.ylabel('Average Wage ($/hour)')
    plt.title('Wage Trends for Top 3 Jobs')
    plt.legend()
    plt.grid(True, alpha=0.3)


    plt.tight_layout()
    plt.savefig('alberta_best_jobs_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSaved visualization to alberta_best_jobs_analysis.png")


    return plt


def main():
    print("Loading Alberta job vacancies data...")
    df = load_and_prepare_data()
    print(f"Data loaded: {df.shape[0]} rows")


    print("\nCalculating job quality metrics...")
    metrics_df = calculate_job_quality_metrics(df)
    print(f"Metrics calculated for {len(metrics_df)} occupations")


    print("\nNormalizing metrics...")
    metrics_norm = normalize_metrics(metrics_df)


    print("\nCalculating composite scores...")
    metrics_scored = calculate_composite_score(metrics_norm)


    # Sort by composite score (descending)
    metrics_scored = metrics_scored.sort_values('Composite_Score', ascending=False)


    print("\n" + "="*80)
    print("BEST JOBS IN ALBERTA - RANKING")
    print("="*80)


    # Display top 15
    display_cols = ['Occupation', 'Current_Wage', 'Wage_Growth_YoY',
                   'Stability_Score', 'Trend_Consistency', 'Composite_Score']


    top_15 = metrics_scored.head(15)
    print(top_15[display_cols].to_string(index=False, float_format='%.3f'))


    print("\n" + "="*80)
    print("DETAILED INSIGHTS")
    print("="*80)


    # Best overall
    if len(metrics_scored) > 0 and not pd.isna(metrics_scored.iloc[0]['Composite_Score']):
        best_job = metrics_scored.iloc[0]
        print(f"\nBEST OVERALL JOB: {best_job['Occupation']}")
        print(f"   Current Wage: ${best_job['Current_Wage']:.2f}/hour")
        print(f"   Wage Growth (YoY): {best_job['Wage_Growth_YoY']:+.2f}%")
        print(f"   Stability Score: {best_job['Stability_Score']:.3f}")
        print(f"   Trend Consistency: {best_job['Trend_Consistency']:.3f}")
        print(f"   Composite Score: {best_job['Composite_Score']:.3f}")


    # Highest wage
    if len(metrics_scored) > 0:
        highest_wage = metrics_scored.loc[metrics_scored['Current_Wage'].idxmax()]
        print(f"\nHIGHEST WAGE: {highest_wage['Occupation']}")
        print(f"   Current Wage: ${highest_wage['Current_Wage']:.2f}/hour")


    # Best growth
    valid_growth = metrics_scored.dropna(subset=['Wage_Growth_YoY'])
    if len(valid_growth) > 0:
        best_growth = valid_growth.loc[valid_growth['Wage_Growth_YoY'].idxmax()]
        print(f"\nBEST WAGE GROWTH: {best_growth['Occupation']}")
        print(f"   Wage Growth (YoY): {best_growth['Wage_Growth_YoY']:+.2f}%")
        print(f"   Current Wage: ${best_growth['Current_Wage']:.2f}/hour")


    # Most stable
    valid_stability = metrics_scored.dropna(subset=['Stability_Score'])
    if len(valid_stability) > 0:
        most_stable = valid_stability.loc[valid_stability['Stability_Score'].idxmax()]
        print(f"\nMOST STABLE: {most_stable['Occupation']}")
        print(f"   Stability Score: {most_stable['Stability_Score']:.3f}")
        print(f"   Current Wage: ${most_stable['Current_Wage']:.2f}/hour")


    # Save results
    output_file = 'alberta_best_jobs_ranking.csv'
    metrics_scored.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")


    # Create visualizations
    visualize_results(metrics_scored)


    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Note: Analysis based on average offered hourly wage data from Statistics Canada")
    print("      Composite score combines: wage level, growth, stability, trend consistency,")
    print("      relative position, and recent momentum (equal weights by default)")


if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def safe_normalize(series):
    """Safely normalize a series to 0-1 range, handling NaN and constant values"""
    if series.isna().all():
        return pd.Series([np.nan] * len(series), index=series.index)

    # Fill NaN with median for normalization calculation, but keep track of original NaN positions
    series_filled = series.fillna(series.median())
    min_val = series_filled.min()
    max_val = series_filled.max()

    if max_val == min_val:
        # All values are the same (or all NaN after filling)
        return pd.Series([0.5] * len(series), index=series.index)
    else:
        normalized = (series_filled - min_val) / (max_val - min_val)
        # Put NaN back where they were originally
        normalized = normalized.where(~series.isna(), np.nan)
        return normalized

def load_and_prepare_data():
    """Load and prepare the Alberta job vacancies data"""
    df = pd.read_csv('alberta_job_vacancies.csv')

    # Filter for Average offered hourly wage (only statistic available)
    df = df[df['Statistics'] == 'Average offered hourly wage'].copy()

    # Convert REF_DATE to datetime
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])

    # Ensure VALUE is numeric
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')

    return df

def analyze_occupation_detailed(df, occupation_name):
    """Analyze a specific occupation in detail"""
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS: {occupation_name}")
    print(f"{'='*60}")

    # Filter data for this occupation
    occ_data = df[df['National Occupational Classification'] == occupation_name].copy()
    occ_data = occ_data.sort_values('REF_DATE')

    if len(occ_data) == 0:
        print(f"No data found for {occupation_name}")
        return None

    print(f"Data points available: {len(occ_data)}")
    print(f"Date range: {occ_data['REF_DATE'].min().strftime('%Y-%m')} to {occ_data['REF_DATE'].max().strftime('%Y-%m')}")

    # Extract wage series and dates - keep them together to ensure same length
    # We'll drop rows where VALUE is NaN, but keep the index alignment
    occ_data_clean = occ_data.dropna(subset=['VALUE']).copy()

    if len(occ_data_clean) == 0:
        print(f"No valid wage data found for {occupation_name}")
        return None

    wages = occ_data_clean['VALUE'].values
    dates = occ_data_clean['REF_DATE'].values

    print(f"Valid data points after cleaning: {len(wages)}")

    # Basic statistics
    current_wage = wages[-1] if len(wages) > 0 else np.nan
    avg_wage = np.mean(wages) if len(wages) > 0 else np.nan
    median_wage = np.median(wages) if len(wages) > 0 else np.nan
    wage_std = np.std(wages) if len(wages) > 0 else np.nan
    min_wage = np.min(wages) if len(wages) > 0 else np.nan
    max_wage = np.max(wages) if len(wages) > 0 else np.nan

    print(f"\nWage Statistics:")
    print(f"  Current wage (latest): ${current_wage:.2f}/hour" if not np.isnan(current_wage) else "  Current wage (latest): N/A")
    print(f"  Average wage: ${avg_wage:.2f}/hour" if not np.isnan(avg_wage) else "  Average wage: N/A")
    print(f"  Median wage: ${median_wage:.2f}/hour" if not np.isnan(median_wage) else "  Median wage: N/A")
    print(f"  Wage range: ${min_wage:.2f} - ${max_wage:.2f}/hour" if not np.isnan(min_wage) and not np.isnan(max_wage) else "  Wage range: N/A")
    print(f"  Standard deviation: ${wage_std:.2f}" if not np.isnan(wage_std) else "  Standard deviation: N/A")

    # Growth metrics
    if len(wages) >= 5:
        wage_yoy = ((wages[-1] - wages[-5]) / wages[-5]) * 100  # YoY change
    else:
        wage_yoy = np.nan

    if len(wages) >= 2:
        wage_qoq = ((wages[-1] - wages[-2]) / wages[-2]) * 100  # QoQ change
    else:
        wage_qoq = np.nan

    print(f"\nGrowth Metrics:")
    print(f"  Year-over-Year change: {wage_yoy:+.2f}%" if not np.isnan(wage_yoy) else "  Year-over-Year change: N/A")
    print(f"  Quarter-over-Quarter change: {wage_qoq:+.2f}%" if not np.isnan(wage_qoq) else "  Quarter-over-Quarter change: N/A")

    # Stability metrics
    if not np.isnan(avg_wage) and avg_wage > 0:
        cv = wage_std / avg_wage  # Coefficient of variation
        stability_score = 1 / (1 + cv)  # Higher = more stable
    else:
        cv = np.nan
        stability_score = np.nan

    print(f"\nStability Metrics:")
    print(f"  Coefficient of Variation: {cv:.3f}" if not np.isnan(cv) else "  Coefficient of Variation: N/A")
    print(f"  Stability Score (0-1): {stability_score:.3f}" if not np.isnan(stability_score) else "  Stability Score: N/A")

    # Trend analysis
    if len(wages) >= 3:
        x = np.arange(len(wages))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, wages, rcond=None)[0]
        predicted = slope * x + intercept
        ss_res = np.sum((wages - predicted) ** 2)
        ss_tot = np.sum((wages - np.mean(wages)) ** 2)
        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
            r_squared = max(0, min(1, r_squared))
        else:
            r_squared = 0

        # Calculate CAGR (Compound Annual Growth Rate)
        if len(wages) >= 4 and wages[0] > 0:
            periods = len(wages) - 1
            years = periods / 4  # Assuming quarterly data
            if years > 0:
                cagr = ((wages[-1] / wages[0]) ** (1/years) - 1) * 100
            else:
                cagr = np.nan
        else:
            cagr = np.nan
    else:
        r_squared = np.nan
        cagr = np.nan

    print(f"\nTrend Metrics:")
    print(f"  Trend Consistency (R-squared): {r_squared:.3f}" if not np.isnan(r_squared) else "  Trend Consistency: N/A")
    print(f"  Compound Annual Growth Rate: {cagr:.2f}%/year" if not np.isnan(cagr) else "  CAGR: N/A")

    # Recent momentum (last 3 periods)
    if len(wages) >= 4:
        recent_3q_change = ((wages[-1] - wages[-4]) / wages[-4]) * 100  # Last 3 quarters
    else:
        recent_3q_change = np.nan

    if len(wages) >= 2:
        recent_1q_change = ((wages[-1] - wages[-2]) / wages[-2]) * 100  # Last quarter
    else:
        recent_1q_change = np.nan

    print(f"\nRecent Momentum:")
    print(f"  Last 1 quarter change: {recent_1q_change:+.2f}%" if not np.isnan(recent_1q_change) else "  Last 1 quarter change: N/A")
    print(f"  Last 3 quarters change: {recent_3q_change:+.2f}%" if not np.isnan(recent_3q_change) else "  Last 3 quarters change: N/A")

    # Volatility analysis (rolling statistics)
    if len(wages) >= 4:
        # 4-quarter rolling average and std
        wages_series = pd.Series(wages)
        rolling_mean = wages_series.rolling(window=4, min_periods=1).mean()
        rolling_std = wages_series.rolling(window=4, min_periods=1).std()

        # Current volatility relative to historical
        current_volatility = rolling_std.iloc[-1] if len(rolling_std) > 0 and not np.isnan(rolling_std.iloc[-1]) else 0
        historical_volatility = rolling_std.iloc[:-1] if len(rolling_std) > 1 else pd.Series([0])
        avg_historical_volatility = np.nanmean(historical_volatility) if len(historical_volatility) > 0 else 0

        if avg_historical_volatility > 0:
            volatility_ratio = current_volatility / avg_historical_volatility
        else:
            volatility_ratio = np.nan
    else:
        volatility_ratio = np.nan

    print(f"\nVolatility Analysis:")
    print(f"  Volatility Ratio (current vs historical): {volatility_ratio:.2f}" if not np.isnan(volatility_ratio) else "  Volatility Ratio: N/A")

    # Create visualization for this occupation
    plt.figure(figsize=(12, 8))

    # Plot 1: Wage over time
    plt.subplot(2, 2, 1)
    if len(dates) > 0 and len(wages) > 0:
        plt.plot(dates, wages, marker='o', linewidth=2, markersize=4)
    plt.title(f'{occupation_name}\nWage Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Wage ($/hour)')
    plt.grid(True, alpha=0.3)

    # Plot 2: Year-over-year change
    plt.subplot(2, 2, 2)
    if len(wages) >= 5:
        yoy_changes = []
        yoy_dates = []
        for i in range(4, len(wages)):
            yoy = ((wages[i] - wages[i-4]) / wages[i-4]) * 100
            yoy_changes.append(yoy)
            yoy_dates.append(dates[i])
        plt.plot(yoy_dates, yoy_changes, marker='s', color='green', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Year-over-Year Wage Change')
        plt.xlabel('Date')
        plt.ylabel('Change (%)')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Insufficient data\nfor YoY calculation',
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Year-over-Year Wage Change')

    # Plot 3: Wage distribution
    plt.subplot(2, 2, 3)
    if len(wages) > 0:
        plt.hist(wages, bins=min(10, len(wages)//2), edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(current_wage, color='red', linestyle='--', linewidth=2, label=f'Current: ${current_wage:.2f}') if not np.isnan(current_wage) else None
        plt.axvline(median_wage, color='green', linestyle='--', linewidth=2, label=f'Median: ${median_wage:.2f}') if not np.isnan(median_wage) else None
    plt.xlabel('Wage ($/hour)')
    plt.ylabel('Frequency')
    plt.title('Wage Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Key metrics summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    metrics_text = f"""
    KEY METRICS SUMMARY
    ───────────────────
    Current Wage:     ${current_wage:.2f}/hr
    Avg Wage:         ${avg_wage:.2f}/hr
    Wage Range:       ${min_wage:.2f} - ${max_wage:.2f}/hr

    Growth:
      YoY Change:     {wage_yoy:+.2f}%
      QoQ Change:     {wage_qoq:+.2f}%

    Stability:
      CV:             {cv:.3f}
      Stability:      {stability_score:.3f}

    Trend:
      R-squared:      {r_squared:.3f}
      CAGR:           {cagr:.2f}%/yr

    Recent:
      1Q Change:      {recent_1q_change:+.2f}%
      3Q Change:      {recent_3q_change:+.2f}%
    """
    # Replace NaN with N/A in the text
    metrics_text = metrics_text.replace('nan', 'N/A').replace('NaN', 'N/A')
    plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.title('Key Metrics Summary')

    plt.tight_layout()
    safe_name = occupation_name.replace(' ', '_').replace('[', '').replace(']', '')
    plt.savefig(f'alberta_{safe_name}_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n[+] Detailed visualization saved as: alberta_{safe_name}_detailed_analysis.png")

    # Return summary dictionary
    return {
        'occupation': occupation_name,
        'current_wage': current_wage,
        'avg_wage': avg_wage,
        'median_wage': median_wage,
        'wage_range_min': min_wage,
        'wage_range_max': max_wage,
        'wage_std': wage_std,
        'wage_yoy': wage_yoy,
        'wage_qoq': wage_qoq,
        'cv': cv,
        'stability_score': stability_score,
        'r_squared': r_squared,
        'cagr': cagr,
        'recent_1q_change': recent_1q_change,
        'recent_3q_change': recent_3q_change,
        'volatility_ratio': volatility_ratio,
        'data_points': len(wages),
        'start_date': dates[0] if len(dates) > 0 else None,
        'end_date': dates[-1] if len(dates) > 0 else None
    }

def compare_occupations(df, occupation_list):
    """Compare multiple occupations side by side"""
    print(f"\n{'='*80}")
    print(f"OCCUPATION COMPARISON ANALYSIS")
    print(f"{'='*80}")

    results = []
    for occupation in occupation_list:
        result = analyze_occupation_detailed(df, occupation)
        if result:
            results.append(result)

    if not results:
        print("No valid occupation data found for comparison")
        return None

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)

    # Select key metrics for comparison
    comparison_cols = ['occupation', 'current_wage', 'wage_yoy', 'stability_score',
                      'r_squared', 'cagr', 'data_points']
    comparison_df = comparison_df[comparison_cols]

    print(f"\n{'='*80}")
    print(f"OCCUPATION COMPARISON TABLE")
    print(f"{'='*80}")
    # Format the dataframe for better display
    formatted_df = comparison_df.copy()
    for col in ['current_wage', 'wage_yoy', 'stability_score', 'r_squared', 'cagr']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f'{x:.2f}' if not pd.isna(x) else 'N/A')
    print(formatted_df.to_string(index=False))

    # Create comparison visualization
    plt.figure(figsize=(16, 12))

    # Plot 1: Current wages
    plt.subplot(2, 3, 1)
    y_pos = np.arange(len(comparison_df))
    current_wages = comparison_df['current_wage'].fillna(0)  # Fill NaN with 0 for plotting
    plt.barh(y_pos, current_wages, color='steelblue', alpha=0.8)
    plt.yticks(y_pos, comparison_df['occupation'])
    plt.xlabel('Current Wage ($/hour)')
    plt.title('Current Wage Levels')
    plt.grid(True, alpha=0.3, axis='x')

    # Plot 2: Wage growth (YoY)
    plt.subplot(2, 3, 2)
    wage_yoy = comparison_df['wage_yoy'].fillna(0)  # Fill NaN with 0 for plotting
    colors = ['red' if x < 0 else 'green' for x in wage_yoy]
    plt.barh(y_pos, wage_yoy, color=colors, alpha=0.8)
    plt.yticks(y_pos, comparison_df['occupation'])
    plt.xlabel('YoY Wage Growth (%)')
    plt.title('Year-over-Year Wage Growth')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3, axis='x')

    # Plot 3: Stability scores
    plt.subplot(2, 3, 3)
    stability_scores = comparison_df['stability_score'].fillna(0)  # Fill NaN with 0 for plotting
    plt.barh(y_pos, stability_scores, color='purple', alpha=0.8)
    plt.yticks(y_pos, comparison_df['occupation'])
    plt.xlabel('Stability Score (0-1)')
    plt.title('Wage Stability')
    plt.grid(True, alpha=0.3, axis='x')

    # Plot 4: Trend consistency (R-squared)
    plt.subplot(2, 3, 4)
    r_squared_vals = comparison_df['r_squared'].fillna(0)  # Fill NaN with 0 for plotting
    plt.barh(y_pos, r_squared_vals, color='orange', alpha=0.8)
    plt.yticks(y_pos, comparison_df['occupation'])
    plt.xlabel('Trend Consistency (R²)')
    plt.title('Wage Trend Reliability')
    plt.grid(True, alpha=0.3, axis='x')

    # Plot 5: CAGR
    plt.subplot(2, 3, 5)
    cagr_vals = comparison_df['cagr'].fillna(0)  # Fill NaN with 0 for plotting
    colors = ['red' if x < 0 else 'green' for x in cagr_vals]
    plt.barh(y_pos, cagr_vals, color=colors, alpha=0.8)
    plt.yticks(y_pos, comparison_df['occupation'])
    plt.xlabel('CAGR (%/year)')
    plt.title('Compound Annual Growth Rate')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3, axis='x')

    # Plot 6: Composite score (equal weights of normalized metrics)
    plt.subplot(2, 3, 6)
    # Normalize metrics for composite score using safe normalization
    norm_wage = safe_normalize(comparison_df['current_wage'])
    norm_growth = safe_normalize(comparison_df['wage_yoy'])
    norm_stability = safe_normalize(comparison_df['stability_score'])  # Already 0-1 range when valid
    norm_trend = safe_normalize(comparison_df['r_squared'])  # Already 0-1 range when valid
    norm_cagr = safe_normalize(comparison_df['cagr'])

    # Calculate composite score as average of available normalized metrics
    norm_arrays = [norm_wage, norm_growth, norm_stability, norm_trend, norm_cagr]
    norm_names = ['wage', 'growth', 'stability', 'trend', 'cagr']

    composite_scores = []
    for i in range(len(comparison_df)):
        scores = []
        for j, norm_array in enumerate(norm_arrays):
            val = norm_array.iloc[i]
            if not np.isnan(val):
                scores.append(val)
        if len(scores) > 0:
            composite_scores.append(np.mean(scores))
        else:
            composite_scores.append(np.nan)

    comparison_df['composite_score'] = composite_scores

    # Plot composite scores
    composite_for_plot = comparison_df['composite_score'].fillna(0)  # Fill NaN with 0 for plotting
    plt.barh(y_pos, composite_for_plot, color='darkblue', alpha=0.8)
    plt.yticks(y_pos, comparison_df['occupation'])
    plt.xlabel('Composite Score (0-1)')
    plt.title('Overall Job Quality Score')
    plt.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('alberta_occupation_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n[+] Occupation comparison visualization saved as: alberta_occupation_comparison.png")

    # Save detailed results
    comparison_df.to_csv('alberta_occupation_detailed_comparison.csv', index=False)
    print(f"[+] Detailed comparison data saved as: alberta_occupation_detailed_comparison.csv")

    return comparison_df

def main():
    print("Loading Alberta job vacancies data...")
    df = load_and_prepare_data()
    print(f"Data loaded: {df.shape[0]} rows")

    # Show available occupations
    occupations = df['National Occupational Classification'].unique()
    print(f"\nAvailable occupations ({len(occupations)}):")
    for i, occ in enumerate(occupations, 1):
        print(f"  {i}. {occ}")

    # Focus on Healthcare and Tech as requested
    healthcare_tech_occupations = [
        occ for occ in occupations
        if 'Health' in occ or 'Natural and applied sciences' in occ
    ]

    print(f"\nSelected for detailed analysis (Healthcare & Tech):")
    for occ in healthcare_tech_occupations:
        print(f"  - {occ}")

    if healthcare_tech_occupations:
        

        # Compare occupations
        print(f"\n{'='*60}")
        print(f"RUNNING COMPARATIVE ANALYSIS")
        print(f"{'='*60}")

        comparison_results = compare_occupations(df, healthcare_tech_occupations)

        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print("Generated files:")
        print("  - Individual occupation detailed analysis plots")
        print("  - alberta_occupation_comparison.png (comparative visualization)")
        print("  - alberta_occupation_detailed_comparison.csv (detailed data)")

        # Show top insights
        if comparison_results is not None and len(comparison_results) > 0:
            print(f"\n[KEY INSIGHTS]:")

            # Highest wage
            valid_wage = comparison_results.dropna(subset=['current_wage'])
            if len(valid_wage) > 0:
                highest_wage = valid_wage.loc[valid_wage['current_wage'].idxmax()]
                print(f"  [HIGH] Highest Wage: {highest_wage['occupation']} (${highest_wage['current_wage']:.2f}/hr)")
            else:
                print(f"  [HIGH] Highest Wage: N/A")

            # Best growth
            valid_growth = comparison_results.dropna(subset=['wage_yoy'])
            if len(valid_growth) > 0:
                best_growth = valid_growth.loc[valid_growth['wage_yoy'].idxmax()]
                print(f"  [GROW] Best Growth: {best_growth['occupation']} ({best_growth['wage_yoy']:+.2f}% YoY)")
            else:
                print(f"  [GROW] Best Growth: N/A")

            # Most stable
            valid_stable = comparison_results.dropna(subset=['stability_score'])
            if len(valid_stable) > 0:
                most_stable = valid_stable.loc[valid_stable['stability_score'].idxmax()]
                print(f"  [STAB] Most Stable: {most_stable['occupation']} (stability: {most_stable['stability_score']:.3f})")
            else:
                print(f"  [STAB] Most Stable: N/A")

            # Best overall
            valid_overall = comparison_results.dropna(subset=['composite_score'])
            if len(valid_overall) > 0:
                best_overall = valid_overall.loc[valid_overall['composite_score'].idxmax()]
                print(f"  [BEST] Best Overall: {best_overall['occupation']} (score: {best_overall['composite_score']:.3f})")
            else:
                print(f"  [BEST] Best Overall: N/A")
    else:
        print("No Healthcare or Tech occupations found in the dataset")

if __name__ == "__main__":
    main()

plt.show()
