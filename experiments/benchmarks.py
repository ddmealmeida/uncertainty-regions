import numpy as np
import pandas as pd
from itertools import chain
from sklearn import datasets
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from baselines.lpr import run_lpr_baseline
from baselines.slice_finder import run_sf_baseline
from eval_pipeline import run_subgroup_discovery, run_hierarchical_clustering, filter_redundant_subgroups
from utils import calculate_coverage_metrics, convert_binary_columns


# Add datasets to the list below, and run the pipeline for each one.
dataset_list = []
dataset_dict = {}
# Moons Dataset
X, y = datasets.make_moons(1000, noise=0.05, random_state=13)
dataset_list.append({'dataset': 'moons',
                     'X': pd.DataFrame(X, columns=['x1', 'x2']),
                     'y': y})

# Iris Dataset
iris = datasets.load_iris()
X = iris.data
y_multi = iris.target
y = (y_multi == 2).astype(int)
dataset_list.append({'dataset': 'iris',
                     'X': pd.DataFrame(X, columns=iris.feature_names),
                     'y': y})

# Wine Quality Dataset
wine_quality = fetch_ucirepo(id=186)
# data (as pandas dataframes)
X = wine_quality.data.features
y = (wine_quality.data.targets > 5).astype(int)
dataset_list.append({'dataset': 'wine',
                     'X': X,
                     'y': y['quality']})

# Student Performance Dataset
student_performance = fetch_ucirepo(id=320)
# data (as pandas dataframes)
X = student_performance.data.features
X = convert_binary_columns(X)
# Keep only numeric columns in X
X = X.select_dtypes(include=np.number)
y = student_performance.data.targets
# Good, Very Good or Excellent Performance
y = (y['G3'] >= 14).astype(int)
dataset_list.append({'dataset': 'student_performance',
                     'X': X,
                     'y': y})

# Diabetes CDC Database
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
# data (as pandas dataframes)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets
dataset_list.append({'dataset': 'diabetes',
                     'X': X,
                     'y': y['Diabetes_binary']})

results_df = pd.DataFrame()
for d in dataset_list:
    lr = LogisticRegression(random_state=57, solver='sag')
    lr.fit(d['X'], d['y'])
    y_pred = lr.predict(d['X'])
    y_prob = lr.predict_proba(d['X'])[:, 1]
    prediction_errors = abs(y_prob - d['y'])
    X_sd = pd.concat([d['X'], pd.Series(prediction_errors, name='errors')],
                     axis=1)
    # Identifying samples with high error
    high_error = X_sd['errors'] > 0.5

    # Run the proposed pipeline
    df = run_subgroup_discovery(X_sd.drop(columns='errors'), X_sd['errors'], a=0.5, result_size=10)
    ac = run_hierarchical_clustering(df)
    distance_threshold = 0.5
    df = filter_redundant_subgroups(df, ac, distance_threshold)

    # Run baseline 1 - LPRs
    df_regras = run_lpr_baseline(X_sd, high_error, len(df))

    # Run baseline 2 - Slice Finder
    slices, slice_cov = run_sf_baseline(lr, X_sd, d['y'], len(df))

    # Calculate metrics for main subgroups
    cumulative_coverage, cumulative_precision = calculate_coverage_metrics(df, high_error)
    # Calculate metrics for baseline 1 (LPR)
    cumulative_coverage_lpr, cumulative_precision_lpr = calculate_coverage_metrics(df_regras, high_error)
    # Calculate metrics for baseline 2 (Slice Finder)
    cumulative_coverage_sf, cumulative_precision_sf = calculate_coverage_metrics(slices, high_error, slice_cov)

    # Save all in a dataframe
    df_plot = pd.DataFrame({"Number of Groups": chain(range(1, len(cumulative_coverage) + 1),
                                                      range(1, len(cumulative_coverage_lpr) + 1),
                                                      range(1, len(cumulative_coverage_sf) + 1)),
                            "Recall": np.concatenate((cumulative_coverage, cumulative_coverage_lpr,
                                                      cumulative_coverage_sf)),
                            "Precision": np.concatenate((cumulative_precision, cumulative_precision_lpr,
                                                         cumulative_precision_sf)),
                            "Strategy": ['Proposed Pipeline'] * len(cumulative_coverage) +
                                        ['Baseline_LPR'] * len(cumulative_coverage_lpr) +
                                        ['Baseline_SF'] * len(cumulative_coverage_sf),
                            "Dataset": d["dataset"]})
    results_df = pd.concat([results_df, df_plot], axis=0)

# Processing the final results
results_df['f1-Score'] = 2 * (results_df['Precision'] * results_df['Recall']) / (results_df['Precision'] + results_df['Recall'])

# Create a ranking for each combination of Dataset and Number of Groups
def get_rank(group):
    # Sort by f1-Score in descending order and assign ranks
    return group['f1-Score'].rank(ascending=False, method='min')

# Group by Dataset and Number of Groups, then calculate ranks
ranked_results = results_df.copy()
ranked_results['Rank'] = ranked_results.groupby(['Dataset', 'Number of Groups'])['f1-Score'].rank(ascending=False,
                                                                                                  method='min')

# Optional: Create a more readable view of the results
ranked_view = ranked_results.pivot_table(
    index=['Dataset', 'Number of Groups'],
    columns='Strategy',
    values=['f1-Score', 'Rank']
).round(3)

final_view = ranked_results.groupby(['Dataset', 'Strategy'])['Rank'].mean().reset_index()
final_view = final_view.pivot_table(
    index=['Dataset'],
    columns='Strategy',
    values='Rank').reset_index()
final_view.columns.name = None