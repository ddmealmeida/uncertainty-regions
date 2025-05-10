import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from itertools import chain
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from baselines.lpr import plot_lprs, run_lpr_baseline
from baselines.slice_finder import plot_slices, run_sf_baseline
from eval_pipeline import (run_subgroup_discovery, run_hierarchical_clustering, filter_redundant_subgroups,
                           plot_subgroups)


# import some data to play with
X, y = datasets.make_moons(1000, noise=0.05, random_state=13)
lr = LogisticRegression(random_state=57)
lr.fit(X, y)
y_pred = lr.predict(X)
y_prob = lr.predict_proba(X)[:, 1]
prediction_errors = abs(y_prob - y)
X_sd = pd.DataFrame(np.concatenate([X, prediction_errors.reshape(-1, 1)],
                                   axis=1),
                    columns=['x1', 'x2', 'errors'])
# Identifying samples with high error
high_error = X_sd['errors'] > 0.5

plot_df = pd.DataFrame(np.concatenate((X, y[:, np.newaxis],
                                       y_pred[:, np.newaxis]),
                                      axis=1),
                       columns=['x1', 'x2', 'actual class', 'predicted class'])
sns.scatterplot(data=plot_df, x='x1', y='x2', hue='predicted class', style='actual class',
                legend='full')
plt.tight_layout()
plt.show()

# Run the subgroup discovery using Beam Search as the search algorithm, using the custom quality function
# defined above
df = run_subgroup_discovery(X_sd.drop(columns='errors'), X_sd['errors'], a=0.5, result_size=10)

# Reducing the redundancy in the subgroups mined, by filtering out similar subgroups
ac = run_hierarchical_clustering(df)
distance_threshold = 0.5
df = filter_redundant_subgroups(df, ac, distance_threshold)

# Plot the subgroups in a 2d scatterplot
plt.figure()
plot_subgroups(pd.DataFrame(X, columns=['x1', 'x2']),
               'x1', 'x2',
               y_pred,
               df.loc[:, ['subgroup', 'mean_sg', 'mean_dataset']])
plt.show()

# Run the baseline 1 (inspired in LPRs, using a decision tree)
df_regras = run_lpr_baseline(X_sd, high_error, 7)

# Plot the baseline subgroups for a better understanding
plt.figure()
plot_lprs(pd.DataFrame(X, columns=['x1', 'x2']),
          'x1', 'x2',
          y_pred,
          df_regras.loc[:, ['lpr_dict', 'mean_sg', 'mean_dataset']],
          plot_mean_error=False)
plt.show()

# Run the baseline 2 (used in the Slice Finder framework, based on lattices)
slices, slice_cov = run_sf_baseline(lr, X_sd, y, 7)

# Plot the baseline subgroups for a better understanding
plt.figure()
plot_slices(pd.DataFrame(X, columns=['x1', 'x2']),
            'x1', 'x2',
            y_pred,
            slices,
            plot_mean_error=False)
plt.show()

# Quantitative analysis
# Checking how many were covered by each number of subgroups found
coverage = np.full((len(X), ), False)
cumulative_coverage = []
cumulative_precision = []
for n_subgroup in range(len(df)):
    coverage = coverage | df.iloc[n_subgroup]['covered']
    new_coverage = sum(coverage[high_error])/sum(high_error)
    cumulative_coverage.append(new_coverage)
    new_precision = sum(coverage[high_error]) / sum(coverage)
    cumulative_precision.append(new_precision)

# Checking how many were covered by each number of baseline 1 subgroups found
coverage_lpr = np.full((len(X), ), False)
cumulative_coverage_lpr = []
cumulative_precision_lpr = []
for n_lpr in range(len(df_regras)):
    coverage_lpr = coverage_lpr | df_regras.iloc[n_lpr]['covered']
    new_coverage_lpr = sum(coverage_lpr[high_error])/sum(high_error)
    cumulative_coverage_lpr.append(new_coverage_lpr)
    new_precision_lpr = sum(coverage_lpr[high_error]) / sum(coverage_lpr)
    cumulative_precision_lpr.append(new_precision_lpr)

# Checking how many were covered by each number of baseline 2 subgroups found
coverage_sf = np.full((len(X), ), False)
cumulative_coverage_sf = []
cumulative_precision_sf = []
for n_slices in range(len(slices)):
    coverage_sf = coverage_sf | slice_cov[n_slices]
    new_coverage_sf = sum(coverage_sf[high_error])/sum(high_error)
    cumulative_coverage_sf.append(new_coverage_sf)
    new_precision_sf = sum(coverage_sf[high_error]) / sum(coverage_sf)
    cumulative_precision_sf.append(new_precision_sf)

df_plot = pd.DataFrame({"Number of Groups": chain(range(1, len(cumulative_coverage) + 1),
                                                  range(1, len(cumulative_coverage_lpr) + 1),
                                                  range(1, len(cumulative_coverage_sf) + 1)),
                        "Recall": np.concatenate((cumulative_coverage, cumulative_coverage_lpr, 
                                                  cumulative_coverage_sf)),
                        "Precision": np.concatenate((cumulative_precision, cumulative_precision_lpr, 
                                                     cumulative_precision_sf)),
                        "Strategy": ['Proposed Pipeline']*len(cumulative_coverage) +
                                    ['Baseline_LPR']*len(cumulative_coverage_lpr) +
                                    ['Baseline_SF']*len(cumulative_coverage_sf)})

# Plot two line plots with the cumulative coverage and precision of the high error samples
fig, axes = plt.subplots(2, 1, sharex='col')
sns.lineplot(data=df_plot,
             x="Number of Groups",
             y="Precision",
             hue='Strategy',
             ax=axes[0])
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Plot a simple line plot with the number of subgroups needed to cover the high error samples
sns.lineplot(data=df_plot,
             x="Number of Groups",
             y="Recall",
             hue='Strategy',
             ax=axes[1])
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
axes[0].set_title("Cumulative Precision and Recall on High Error Samples")
fig.show()

# Ao invés de plotar o número de subgrupos pra atingir x% de cobertura, talvez seja melhor plotar o percentual total de
# amostras selecionadas X cobertura das de alto erro

# Checking the percentage of high error covered by each number of subgroups found
coverage = np.full((len(X), ), False)
cumulative_coverage = []
total_coverage = []
for n_subgroup in range(len(df)):
    coverage = coverage | df.iloc[n_subgroup]['covered']
    new_coverage = sum(coverage[high_error])/sum(high_error)
    cumulative_coverage.append(new_coverage)
    total_coverage.append(sum(coverage) / len(coverage))

# Checking the percentage of high error covered by each number of LPRs found
coverage_lpr = np.full((len(X), ), False)
cumulative_coverage_lpr = []
total_coverage_lpr = []
for n_lpr in range(len(df_regras)):
    coverage_lpr = coverage_lpr | df_regras.iloc[n_lpr]['covered']
    new_coverage_lpr = sum(coverage_lpr[high_error])/sum(high_error)
    cumulative_coverage_lpr.append(new_coverage_lpr)
    total_coverage_lpr.append(sum(coverage_lpr)/len(coverage_lpr))
    
# Checking the percentage of high error covered by each number of slices found
coverage_sf = np.full((len(X), ), False)
cumulative_coverage_sf = []
total_coverage_sf = []
for n_slices in range(len(slices)):
    coverage_sf = coverage_sf | slice_cov[n_slices]
    new_coverage_sf = sum(coverage_sf[high_error])/sum(high_error)
    cumulative_coverage_sf.append(new_coverage_sf)
    total_coverage_sf.append(sum(coverage_sf)/len(coverage_sf))


df_plot2 = pd.DataFrame({"Percentage of Samples Selected": np.concatenate((total_coverage, total_coverage_lpr,
                                                                           total_coverage_sf)),
                        "High Error Samples Coverage": np.concatenate((cumulative_coverage, cumulative_coverage_lpr,
                                                                       cumulative_coverage_sf)),
                        "Strategy": ['Proposed Pipeline']*len(cumulative_coverage) +
                                    ['Baseline_LPR']*len(cumulative_coverage_lpr) +
                                    ['Baseline_SF']*len(cumulative_coverage_sf)})

# Plot a simple line plot with the number of subgroups needed to cover the high error samples
ax = sns.lineplot(data=df_plot2,
                  x="Percentage of Samples Selected",
                  y="High Error Samples Coverage",
                  hue='Strategy')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Cumulative Coverage of High Error Samples")
plt.show()
