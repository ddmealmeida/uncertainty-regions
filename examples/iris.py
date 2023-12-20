import numpy as np
import pandas as pd
import pysubgroup as ps
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections.abc import Iterable
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from venn import venn


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
            ax.text(rule1_lowerbound, rule2_lowerbound, round(subgroup.mean_sg, 4), fontsize=8)
            ax.text(rule1_upperbound, rule2_upperbound, round(subgroup.mean_dataset, 4), fontsize=8)
        else:
            ax.add_patch(plt.Rectangle((rule2_lowerbound - delta, rule1_lowerbound - delta),
                                       width=rule2_upperbound - rule2_lowerbound + 2*delta,
                                       height=rule1_upperbound - rule1_lowerbound + 2*delta,
                                       fill=False, edgecolor=color, linewidth=1))
            ax.text(rule2_lowerbound, rule1_lowerbound, round(subgroup.mean_sg, 4), fontsize=8)
            ax.text(rule2_upperbound, rule1_upperbound, round(subgroup.mean_dataset, 4), fontsize=8)
    return ax


# import some data to play with
iris = datasets.load_iris()
X = iris.data
y_multi = iris.target

model_list = []
errors_df = pd.DataFrame()
for c in range(3):
    # build a binary classifier to identify each class
    y = (y_multi == c).astype(int)
    rf = RandomForestClassifier(random_state=57)
    rf.fit(X, y)
    model_list.append(rf)
    y_prob = rf.predict_proba(X)[:, 1]
    prediction_error = abs(y - y_prob)
    errors_df = pd.concat([errors_df, pd.Series(prediction_error, name=c)], axis=1)


# Plot a boxplot with seaborn, using target names as labels
only_errors = errors_df.loc[:, range(3)]
only_errors.columns = iris.target_names
only_errors = only_errors.melt(var_name='class', value_name='error')
plt.figure()
sns.boxplot(x='class', y='error', data=only_errors)
plt.show()
# Print the average error for each class
print(only_errors.groupby('class')['error'].mean())

# Run the subgroup discovery for each model using Beam Search as the search algorithm, using the custom quality function
# defined above
X_sd = pd.concat([pd.DataFrame(X, columns=iris['feature_names']), errors_df], axis=1)
df_dict = {}
for class_of_interest in range(3):
    num_target = ps.NumericTarget(class_of_interest)
    searchspace = ps.create_selectors(X_sd, ignore=range(3))
    task = ps.SubgroupDiscoveryTask(
        X_sd,
        num_target,
        searchspace,
        result_set_size=20,
        depth=2,
        qf=ps.StandardQFNumeric(a=0.5))
    print('Mining relevant subgroups...')
    result = ps.BeamSearch().execute(task=task)
    df_subgroup = result.to_dataframe()
    df_subgroup['covered'] = df_subgroup['subgroup'].apply(lambda x: x.covers(X_sd))
    df_subgroup['class'] = class_of_interest
    df_dict[str(class_of_interest)] = df_subgroup.copy()

# Writing a LaTeX table with the subgroups found for one of the models
df_print = df_dict['1'].copy()
df_print.loc[:, 'subgroup'] = df_print['subgroup'].apply(lambda x: x.__str__().replace('sepal width (cm)', 'sw'))
df_print.loc[:, 'subgroup'] = df_print['subgroup'].apply(lambda x: x.__str__().replace('sepal length (cm)', 'sl'))
df_print.loc[:, 'subgroup'] = df_print['subgroup'].apply(lambda x: x.__str__().replace('petal width (cm)', 'pw'))
df_print.loc[:, 'subgroup'] = df_print['subgroup'].apply(lambda x: x.__str__().replace('petal length (cm)', 'pl'))
print(df_print.to_latex(columns=['quality', 'subgroup', 'size_sg', 'mean_sg'],
                        header=['Quality', 'Subgroup', 'Size', 'Average Error'],
                        index=False,
                        float_format="{:.2f}".format))

# Reducing the redundancy in the subgroups mined, by filtering out similar subgroups
for target_class, df in df_dict.items():
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
    df['subgroup_short'] = df['subgroup'].apply(lambda x: x.__str__().replace('sepal width (cm)', 'sw'))
    df.loc[:, 'subgroup_short'] = df['subgroup_short'].apply(lambda x: x.__str__().replace('sepal length (cm)', 'sl'))
    df.loc[:, 'subgroup_short'] = df['subgroup_short'].apply(lambda x: x.__str__().replace('petal width (cm)', 'pw'))
    df.loc[:, 'subgroup_short'] = df['subgroup_short'].apply(lambda x: x.__str__().replace('petal length (cm)', 'pl'))

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
    distance_threshold = 0.25
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
    subgroup_filter = ~df_dict[target_class]['subgroup'].isin(subgroup_replacements.keys())
    df_dict[target_class] = df_dict[target_class].loc[subgroup_filter]


# Concatenate the subgroups found for each model into a single dataframe
df_subgroup = pd.concat(df_dict, ignore_index=True)
df_subgroup = df_subgroup.replace({'class': {0: 'setosa',
                                             1: 'versicolor',
                                             2: 'virginica'}})

# Plot the subgroups in a 2d scatterplot
plt.figure()
plot_subgroups(pd.DataFrame(X, columns=iris.feature_names),
               'petal width (cm)', 'petal length (cm)',
               [iris.target_names[x] for x in y_multi],
               df_subgroup.loc[[6, 18, 25], ['subgroup', 'mean_sg', 'mean_dataset']])
plt.show()

# Plot a Venn Diagram to compare which subgroups were identified as hard or easy for each of the models
set_list = []
set_dict = {}
for classe in df_subgroup['class'].unique():
    set_list.append(set(df_subgroup.loc[df_subgroup['class'] == classe, 'subgroup']))
    set_dict[classe] = set(df_subgroup.loc[df_subgroup['class'] == classe, 'subgroup'])

venn(set_dict)
plt.show()

# Select only the subgroups that were identified as hard for the versicolor or virginica classes, but not for setosa
set_dict = {}
for classe in df_subgroup['class'].unique():
    set_dict[classe] = set(df_subgroup.loc[df_subgroup['class'] == classe, 'subgroup'])
regras_interesse = set_dict['versicolor'].union(set_dict['virginica']) - set_dict['setosa']
df_interesse = df_subgroup.loc[df_subgroup['subgroup'].isin(regras_interesse), ['subgroup', 'covered']]
df_interesse.drop_duplicates(subset='subgroup', inplace=True)
