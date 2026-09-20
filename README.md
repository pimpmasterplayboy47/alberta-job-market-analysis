# Alberta Job Market Analysis

A Python data-analysis project exploring job vacancies, wages, and job quality across occupations in Alberta using Statistics Canada data.

The analysis includes:

* Average offered hourly wage trends in Alberta
* Wage comparisons across occupations
* Year-over-year wage changes
* Descriptive wage statistics
* Job-quality metrics
* A composite job-quality score
* Detailed analysis of selected occupations
* Visualizations and CSV datasets containing the analysis results

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

Generated analysis files are excluded from the Git repository using `.gitignore`.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Alberta-Job-Analysis.git
cd Alberta-Job-Analysis
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Data

The project uses Alberta job vacancy data from Statistics Canada.

The primary dataset is:

```text
alberta_job_vacancies.csv
```

The analysis filters the data to examine measures such as average offered hourly wages and organizes the information by occupation and time period.

## Running the Analysis

Run the main Python script:

```bash
python main.py
```

The program performs the analysis and generates visualizations and CSV files containing the results.

## Output

The analysis produces several outputs, including:

* `alberta_avg_wage_trend.png` — Alberta average offered wage trend
* `alberta_wage_by_occupation.png` — Wage comparison across occupations
* `alberta_best_jobs_analysis.png` — Job-quality analysis
* `alberta_occupation_comparison.png` — Comparison of selected occupations
* `alberta_occupation_detailed_analysis.png` — Detailed occupation analysis
* `alberta_job_vacancies_processed.csv` — Processed wage data
* `alberta_wage_by_occupation_pivot.csv` — Wage data organized by occupation and date
* `alberta_best_jobs_ranking.csv` — Job-quality ranking results
* `alberta_occupation_detailed_comparison.csv` — Detailed occupation comparison data

These files are generated when the program runs and are excluded from version control where appropriate.

## Purpose

The goal of this project is to practice data cleaning, statistical analysis, visualization, and interpretation using real-world Canadian labour-market data.

It also demonstrates how Python can be used to transform a large dataset into useful information about Alberta's job market.
