import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
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