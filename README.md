# Alberta Job Market Analysis

A Python data-analysis project exploring job vacancies, wages, and job quality across occupations in Alberta using Statistics Canada data.

## Overview

This project analyzes Alberta job vacancy and wage data to identify trends in offered wages and compare different occupations based on several job-quality metrics.

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

## Project Structure

```text
Alberta-Job-Analysis/
├── main.py
├── alberta_job_vacancies.csv
├── requirements.txt
├── README.md
└── .gitignore
```

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

## Future Improvements

Possible future improvements include:

* Adding additional labour-market indicators
* Improving the job-quality scoring methodology
* Adding more occupations to the detailed comparison
* Creating an interactive dashboard
* Automating data updates when new Statistics Canada data becomes available
* Organizing generated outputs into a dedicated `outputs/` directory


<img width="1000" height="600" alt="alberta_avg_wage_trend" src="https://github.com/user-attachments/assets/073b6a79-dff3-47ed-8969-3d4d58a1fe1d" />

<img width="4200" height="2400" alt="alberta_wage_by_occupation" src="https://github.com/user-attachments/assets/e2119a3d-4409-430f-baf7-9a2201e6cab6" />

<img width="4166" height="2965" alt="alberta_best_jobs_analysis" src="https://github.com/user-attachments/assets/fa4ce980-8429-490f-b141-67f5a6f1d153" />

<img width="3565" height="2402" alt="alberta_Natural_and_applied_sciences_and_related_occupations_2_detailed_analysis" src="https://github.com/user-attachments/assets/17e6c12a-0693-43f6-8667-15785dffecea" />

<img width="3565" height="2402" alt="alberta_Health_occupations_3_detailed_analysis" src="https://github.com/user-attachments/assets/0a8158a7-5f7e-4a60-af59-97a596d5c36f" />

<img width="4819" height="3563" alt="alberta_occupation_comparison" src="https://github.com/user-attachments/assets/24ec6f36-b28e-4d3b-9ad0-90b9358577ce" />
