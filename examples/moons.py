import numpy as np
import pandas as pd
import pysubgroup as ps
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from itertools import combinations, chain
from collections.abc import Iterable
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from baselines.lpr import rules, get_max_min, get_support, summarize_descriptors, plot_lprs, summarize_lprs


def boolean_jaccard(set1, set2):
    return sum(set1 & set2) / sum(set1 | set2)


def sets_jaccard(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_subgroups(data: pd.DataFrame,
                   x_column: str,
                   y_column: str,
                   target: Iterable,
                   subgroups: pd.DataFrame,
                   plot_mean_error: bool = True,
                   ax=None):
    """Plot a 2D scatterplot showing the samples, its classes, and the subgroups passed as parameters
    to the function"""

    if ax is None:
        ax = plt.gca()

    sns.scatterplot(data=data,
                    x=x_column,
                    y=y_column,
                    hue=target,
                    s=20, alpha=0.5, ax=ax)
    # Rectangle displacement, estimated from my head
    delta = 0.02
    for subgroup in subgroups.itertuples(index=False):
        # extract the subgroup limits
        rules = subgroup.subgroup.selectors
        if len(rules) < 2:
            rule = rules[0]
            if isinstance(rule, ps.IntervalSelector):
                rule1_upperbound = rule.upper_bound
                rule1_lowerbound = rule.lower_bound
                rule1_attribute = rule.attribute_name
                if rule1_lowerbound == float("-inf"):
                    rule1_lowerbound = data[rule1_attribute].min()
                if rule1_upperbound == float("inf"):
                    rule1_upperbound = data[rule1_attribute].max()
            if rule1_attribute == x_column:
                rule2_attribute = y_column
            else:
                rule2_attribute = x_column
            rule2_lowerbound = data[rule2_attribute].min()
            rule2_upperbound = data[rule2_attribute].max()
        else:
            rule1, rule2 = subgroup.subgroup.selectors
            if isinstance(rule1, ps.IntervalSelector):
                rule1_upperbound = rule1.upper_bound
                rule1_lowerbound = rule1.lower_bound
                rule1_attribute = rule1.attribute_name
                if rule1_lowerbound == float("-inf"):
                    rule1_lowerbound = data[rule1_attribute].min()
                if rule1_upperbound == float("inf"):
                    rule1_upperbound = data[rule1_attribute].max()
            else:
                raise(NotImplementedError("I still can't deal with non numeric features!"))
            if isinstance(rule2, ps.IntervalSelector):
                rule2_upperbound = rule2.upper_bound
                rule2_lowerbound = rule2.lower_bound
                rule2_attribute = rule2.attribute_name
                if rule2_lowerbound == float("-inf"):
                    rule2_lowerbound = data[rule2_attribute].min()
                if rule2_upperbound == float("inf"):
                    rule2_upperbound = data[rule2_attribute].max()
            else:
                raise(NotImplementedError("I still can't deal with non numeric features!"))
        # draw a red or green rectangle around the region of interest
        if subgroup.mean_sg > subgroup.mean_dataset:
            color = 'red'
        else:
            color = 'green'
        if rule1_attribute == x_column:
            ax.add_patch(plt.Rectangle((rule1_lowerbound - delta, rule2_lowerbound - delta),
                                       width=rule1_upperbound - rule1_lowerbound + 2*delta,
                                       height=rule2_upperbound - rule2_lowerbound + 2*delta,
                                       fill=False, edgecolor=color, linewidth=1))
            if plot_mean_error:
                ax.text(rule1_lowerbound, rule2_lowerbound, round(subgroup.mean_sg, 4), fontsize=8)
            # ax.text(rule1_upperbound, rule2_upperbound, round(subgroup.mean_dataset, 4), fontsize=8)
        else:
            ax.add_patch(plt.Rectangle((rule2_lowerbound - delta, rule1_lowerbound - delta),
                                       width=rule2_upperbound - rule2_lowerbound + 2*delta,
                                       height=rule1_upperbound - rule1_lowerbound + 2*delta,
                                       fill=False, edgecolor=color, linewidth=1))
            if plot_mean_error:
                ax.text(rule2_lowerbound, rule1_lowerbound, round(subgroup.mean_sg, 4), fontsize=8)
            # ax.text(rule2_upperbound, rule1_upperbound, round(subgroup.mean_dataset, 4), fontsize=8)
    return ax


# import some data to play with
X, y = datasets.make_moons(1000, noise=0.05, random_state=13)
lr = LogisticRegression(random_state=57)
lr.fit(X, y)
y_pred = lr.predict(X)
y_prob = lr.predict_proba(X)[:, 1]
prediction_errors = abs(y_prob - y)

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
X_sd = pd.DataFrame(np.concatenate([X, prediction_errors.reshape(-1, 1)],
                                   axis=1),
                    columns=['x1', 'x2', 'errors'])

num_target = ps.NumericTarget('errors')
searchspace = ps.create_selectors(X_sd, ignore='errors')
task = ps.SubgroupDiscoveryTask(
    X_sd,
    num_target,
    searchspace,
    result_set_size=10,
    depth=2,
    qf=ps.StandardQFNumeric(a=0.2))
print('Mining relevant subgroups...')
result = ps.BeamSearch().execute(task=task)
df = result.to_dataframe()
df['covered'] = df['subgroup'].apply(lambda x: x.covers(X_sd))

# Reducing the redundancy in the subgroups mined, by filtering out similar subgroups

# Pre-calculating the Jaccard Index
jaccard_generator1 = (1 - boolean_jaccard(row1, row2) for row1, row2 in combinations(df['covered'], r=2))
jaccard_generator2 = (1 - sets_jaccard(set(row1.selectors),
                                       set(row2.selectors)) for row1, row2 in combinations(df['subgroup'], r=2))
flattened_matrix1 = np.fromiter(jaccard_generator1, dtype=np.float64)
flattened_matrix2 = np.fromiter(jaccard_generator2, dtype=np.float64)
flattened_matrix = np.minimum(flattened_matrix1, flattened_matrix2)

# since flattened_matrix is the flattened upper triangle of the matrix
# we need to expand it.
normal_matrix = distance.squareform(flattened_matrix)
# replacing zeros with ones at the diagonal.
# normal_matrix += np.identity(len(df_interesse['covered']))

# setting distance_threshold=0 ensures we compute the full tree.
ac = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='average')
ac.fit(normal_matrix)

# Make the feature names shorter for the visualization
df['subgroup_short'] = df['subgroup'].apply(lambda x: x.__str__())

# plot the dendrogram
plt.figure()
plot_dendrogram(ac, truncate_mode="level", p=13, orientation='left',
                labels=list(df['subgroup_short']), leaf_font_size=8.)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlim(1, min(ac.distances_))
plt.tight_layout()
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# Set the distance threshold used for filtering the redundant subgroups
distance_threshold = 0.5
n_samples = len(ac.labels_)
dict_nodes = {}  # Save the representative subgroup for each merge
subgroup_replacements = {}  # Save the original and substitute subgroups
for i, dist in enumerate(ac.distances_):
    if dist > distance_threshold:
        break
    if ac.children_[i][0] >= n_samples:
        first_index = dict_nodes[ac.children_[i][0] - n_samples]
    else:
        first_index = ac.children_[i][0]
    if ac.children_[i][1] >= n_samples:
        second_index = dict_nodes[ac.children_[i][1] - n_samples]
    else:
        second_index = ac.children_[i][1]
    if df.loc[first_index, 'quality'] > df.loc[second_index, 'quality']:
        dict_nodes[i] = first_index
        subgroup_replacements[df.loc[second_index, 'subgroup']] = df.loc[first_index, 'subgroup']
    else:
        dict_nodes[i] = second_index
        subgroup_replacements[df.loc[first_index, 'subgroup']] = df.loc[second_index, 'subgroup']

# Print the similar subgroups and their representatives saved in subgroup_replacements
print(subgroup_replacements)
# Drop the similar subgroups from the dataframe
subgroup_filter = ~df['subgroup'].isin(subgroup_replacements.keys())
df = df.loc[subgroup_filter]


# Plot the subgroups in a 2d scatterplot
plt.figure()
plot_subgroups(pd.DataFrame(X, columns=['x1', 'x2']),
               'x1', 'x2',
               y_pred,
               df.loc[:, ['subgroup', 'mean_sg', 'mean_dataset']])
plt.show()

# Run the baseline 1 (inspired in LPRs, using a decision tree) and use the non-leaf nodes as subgroups
df_regras = pd.DataFrame()
support_threshold = 0.05  # As used in the LPR paper
minimum_threshold = 0.18  # 10% greater than the average error
min_samples = int(support_threshold*X_sd.shape[0])
# Rules with only one column
for col in X_sd.columns[:-1]:
    clf = DecisionTreeRegressor(min_samples_leaf=min_samples,
                                min_samples_split=20,
                                max_depth=30,
                                random_state=57).fit(X_sd[[col]], X_sd.iloc[:,-1])  # Fits the tree
    regras = rules(clf, X_sd[[col]])  # Extracts the rules generated by the tree

    df_part = get_max_min(regras, X_sd[[col, 'errors']])
    df_part = get_support(df_part, X_sd[[col, 'errors']])
    filtro1 = df_part['support'] >= support_threshold
    filtro2 = df_part['average_error'] >= minimum_threshold
    df_part = df_part.loc[filtro1 & filtro2]
    df_regras = pd.concat([df_regras, df_part], ignore_index=True)

# Regras com 2 colunas cada
for n,col1 in enumerate(X_sd.columns[:-1]):
    for col2 in X_sd.columns[n + 1 :-1]:
        clf = DecisionTreeRegressor(max_depth=30,
                                    min_samples_split=20,
                                    min_samples_leaf=min_samples,
                                    random_state=57).fit(X_sd[[col1,col2]], X_sd.iloc[:,-1])
        regras = rules(clf, X_sd[[col1,col2]])
        df_part = get_max_min(regras, X_sd[[col1,col2, 'errors']])
        df_part = get_support(df_part, X_sd[[col1,col2, 'errors']])
        filtro1 = df_part['support'] >= support_threshold
        filtro2 = df_part['average_error'] >= minimum_threshold
        df_part = df_part.loc[filtro1 & filtro2]
        df_regras = pd.concat([df_regras,df_part], ignore_index=True)

lpr_desc, lpr_dict = summarize_descriptors(df_regras, X_sd.columns[:-1], 'data')
# adjusting coverage representation for further analyses
df_regras['covered'] = df_regras['coverage'].apply(lambda x: X_sd.index.isin(x))
df_regras['lpr_desc'] = lpr_desc
df_regras['lpr_dict'] = lpr_dict
df_regras['mean_sg'] = 1
df_regras['mean_dataset'] = 0

# Identifying samples with high error
high_error = X_sd['errors'] > 0.5
df_regras = summarize_lprs(df_regras, 7, X_sd.loc[high_error])

# Plot the baseline subgroups for a better understanding
plt.figure()
plot_lprs(pd.DataFrame(X, columns=['x1', 'x2']),
          'x1', 'x2',
          y_pred,
          df_regras.loc[:, ['lpr_dict', 'mean_sg', 'mean_dataset']],
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
    if new_coverage == 1:
        break

# Checking how many were covered by each number of baseline subgroups found
coverage_lpr = np.full((len(X), ), False)
cumulative_coverage_lpr = []
cumulative_precision_lpr = []
for n_lpr in range(len(df_regras)):
    coverage_lpr = coverage_lpr | df_regras.iloc[n_lpr]['covered']
    new_coverage_lpr = sum(coverage_lpr[high_error])/sum(high_error)
    cumulative_coverage_lpr.append(new_coverage_lpr)
    new_precision_lpr = sum(coverage_lpr[high_error]) / sum(coverage_lpr)
    cumulative_precision_lpr.append(new_precision_lpr)
    if new_coverage_lpr == 1:
        break

df_plot = pd.DataFrame({"Number of Groups": chain(range(1, len(cumulative_coverage) + 1),
                                                     range(1, len(cumulative_coverage_lpr) + 1)),
                        "Recall": np.concatenate((cumulative_coverage, cumulative_coverage_lpr)),
                        "Precision": np.concatenate((cumulative_precision, cumulative_precision_lpr)),
                        "Strategy": ['Proposed Pipeline']*len(cumulative_coverage) +
                                    ['Baseline']*len(cumulative_coverage_lpr)})

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

# Checking how many were covered by each number of subgroups found
coverage = np.full((len(X), ), False)
cumulative_coverage = []
total_coverage = []
for n_subgroup in range(len(df)):
    coverage = coverage | df.iloc[n_subgroup]['covered']
    new_coverage = sum(coverage[high_error])/sum(high_error)
    cumulative_coverage.append(new_coverage)
    total_coverage.append(sum(coverage) / len(coverage))
    # if new_coverage == 1:
    #     break

# Checking how many were covered by each percentage of samples covered
coverage_lpr = np.full((len(X), ), False)
cumulative_coverage_lpr = []
total_coverage_lpr = []
for n_lpr in range(len(df_regras)):
    coverage_lpr = coverage_lpr | df_regras.iloc[n_lpr]['covered']
    new_coverage_lpr = sum(coverage_lpr[high_error])/sum(high_error)
    cumulative_coverage_lpr.append(new_coverage_lpr)
    total_coverage_lpr.append(sum(coverage_lpr)/len(coverage_lpr))
    # if new_coverage_km == 1:
    #     break

df_plot2 = pd.DataFrame({"Percentage of Samples Selected": np.concatenate((total_coverage, total_coverage_lpr)),
                        "High Error Samples Coverage": np.concatenate((cumulative_coverage, cumulative_coverage_lpr)),
                        "Strategy": ['Proposed Pipeline']*len(cumulative_coverage) +
                                    ['Baseline']*len(cumulative_coverage_lpr)})

# Plot a simple line plot with the number of subgroups needed to cover the high error samples
ax = sns.lineplot(data=df_plot2,
                  x="Percentage of Samples Selected",
                  y="High Error Samples Coverage",
                  hue='Strategy')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Cumulative Coverage of High Error Samples")
plt.show()
