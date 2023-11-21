import numpy as np
import pandas as pd
import pysubgroup as ps
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import namedtuple
from collections.abc import Iterable
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from venn import venn

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


# Define a custom quality function for a bidirectional search over the model's errors
class BidirectionalQFNumeric(ps.BoundedInterestingnessMeasure):
    tpl = namedtuple("BidirectionalQFNumeric_parameters", ("size_sg", "mean", "estimate"))
    mean_tpl = tpl

    @staticmethod
    def bidirectional_qf_numeric(a, _, mean_dataset, instances_subgroup, mean_sg):
        return instances_subgroup**a * abs(mean_sg - mean_dataset)

    def __init__(self, a, invert=False, estimator="sum", centroid="mean"):
        self.a = a
        self.invert = invert
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False
        self.required_stat_attrs = ("size_sg", "mean")
        self.agg = np.mean
        self.tpl = BidirectionalQFNumeric.mean_tpl
        self.read_centroid = lambda x: x.mean
        self.estimator = BidirectionalQFNumeric.SummationEstimator(self)

    def calculate_constant_statistics(self, data, target):
        data = self.estimator.get_data(data, target)
        self.all_target_values = data[target.target_variable].to_numpy()
        target_centroid = self.agg(self.all_target_values)
        data_size = len(data)
        self.dataset_statistics = self.tpl(data_size, target_centroid, None)
        self.estimator.calculate_constant_statistics(data, target)
        self.has_constant_statistics = True

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return BidirectionalQFNumeric.bidirectional_qf_numeric(
            self.a,
            dataset.size_sg,
            self.read_centroid(dataset),
            statistics.size_sg,
            self.read_centroid(statistics),
        )

    def calculate_statistics(self, subgroup, target, data, statistics=None):  # pylint: disable=unused-argument
        cover_arr, sg_size = ps.get_cover_array_and_size(
            subgroup, len(self.all_target_values), data
        )
        sg_centroid = 0
        sg_target_values = 0
        if sg_size > 0:
            sg_target_values = self.all_target_values[cover_arr]
            sg_centroid = self.agg(sg_target_values)
            estimate = self.estimator.get_estimate(
                subgroup, sg_size, sg_centroid, cover_arr, sg_target_values
            )
        else:
            estimate = float("-inf")
        return self.tpl(sg_size, sg_centroid, estimate)

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        return float("+inf")

    class SummationEstimator:
        r"""\
        This estimator calculates the optimistic estimate as a hypothetical subgroup\
         which contains only instances with value greater than the dataset mean and\
         is of maximal size.
        .. math::
            oe(sg) = \sum_{x \in sg, T(x)>0} (T(sg) - \mu_0)

        From Florian Lemmerich's Dissertation [section 4.2.2.1, Theorem 2 (page 81)]
        """

        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_centroid = None
            self.target_values_greater_centroid = None

        def get_data(self, data, target):  # pylint: disable=unused-argument
            return data

        def calculate_constant_statistics(self, data, target):  # pylint: disable=unused-argument
            self.indices_greater_centroid = (
                self.qf.all_target_values
                > self.qf.read_centroid(self.qf.dataset_statistics)
            )
            self.target_values_greater_centroid = (
                self.qf.all_target_values
            )  # [self.indices_greater_mean]

        def get_estimate(self, subgroup, sg_size, sg_centroid, cover_arr, _):  # pylint: disable=unused-argument
            larger_than_centroid = self.target_values_greater_centroid[cover_arr][
                self.indices_greater_centroid[cover_arr]
            ]
            size_greater_centroid = len(larger_than_centroid)
            sum_greater_centroid = np.sum(larger_than_centroid)

            return sum_greater_centroid - size_greater_centroid * self.qf.read_centroid(
                self.qf.dataset_statistics
            )


# Run the subgroup discovery for each model using Beam Search as the search algorithm, using the custom quality function
# defined above
X_sd = pd.concat([pd.DataFrame(X, columns=iris['feature_names']), errors_df], axis=1)
print(X_sd.columns)
exit(0)
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
        qf=BidirectionalQFNumeric(a=0.5))
    print('Mining relevant subgroups...')
    result = ps.BeamSearch().execute(task=task)
    df_regras = result.to_dataframe()
    df_regras['covered'] = df_regras['subgroup'].apply(lambda x: x.covers(X_sd))
    df_regras['class'] = class_of_interest
    df_dict[str(class_of_interest)] = df_regras.copy()

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

# Reducing the redundancy in the subgroups mined, by unifying the names of equal coverage subgroup descriptions
df_regras = pd.concat(df_dict, ignore_index=True)
lista_regras = []
lista_intersecoes = []

# First, we create a list of the subgroup descriptions and its coverage sets
for _, linha_regra in tqdm(df_regras.iterrows()):
    perc_igual = 0
    for r in lista_regras:
        perc_igual = sum(linha_regra['covered'] & r['covered']) / sum(linha_regra['covered'] | r['covered'])
        lista_intersecoes.append(perc_igual)
        if perc_igual == 1:
            r['subgroup'].add(linha_regra['subgroup'])
            break
    if perc_igual != 1:
        lista_regras.append({'subgroup': {linha_regra['subgroup']}, 'covered': linha_regra['covered']})

# Then, we create a dictionary to replace all descriptions by the first of them
regras_repetidas = [x['subgroup'] for x in lista_regras if len(x['subgroup']) > 1]
mapa_regras = {}
for conjunto in regras_repetidas:
    primeira_regra = None
    for r in conjunto:
        if primeira_regra is None:
            primeira_regra = r
        else:
            mapa_regras[r] = primeira_regra

df_regras = df_regras.replace({'subgroup': mapa_regras,
                               'class': {0: 'setosa',
                                         1: 'versicolor',
                                         2: 'virginica'}})


# Plot a 2D scatterplot showing the samples, its classes, and the subgroups passed as parameters
# to the function
def plot_subgroups(data: pd.DataFrame,
                   x_column: str,
                   y_column: str,
                   target: Iterable,
                   subgroups: pd.DataFrame,
                   ax=None):
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

plt.figure()
plot_subgroups(pd.DataFrame(X, columns=iris.feature_names),
               'petal width (cm)', 'petal length (cm)',
               [iris.target_names[x] for x in y_multi],
               df_regras.loc[[6, 20, 27, 56], ['subgroup', 'mean_sg', 'mean_dataset']])
plt.show()

# Plot a Venn Diagram to compare which subgroups were identified as hard or easy for each of the models
set_list = []
set_dict = {}
for classe in df_regras['class'].unique():
    set_list.append(set(df_regras.loc[df_regras['class'] == classe, 'subgroup']))
    set_dict[classe] = set(df_regras.loc[df_regras['class'] == classe, 'subgroup'])

venn(set_dict)
plt.show()

# Run an AgglomerativeClustering procedure to generate a dendrogram based on the Jaccard index between the coverage sets
# of the subgroups found for class versicolor and virginica but not for setosa
set_dict = {}
for classe in df_regras['class'].unique():
    set_dict[classe] = set(df_regras.loc[df_regras['class'] == classe, 'subgroup'])
regras_interesse = set_dict['versicolor'].union(set_dict['virginica']) - set_dict['setosa']
df_interesse = df_regras.loc[df_regras['subgroup'].isin(regras_interesse), ['subgroup', 'covered']]
df_interesse.drop_duplicates(subset='subgroup', inplace=True)


# Pre-calculating the Jaccard Index
def boolean_jaccard(set1, set2):
    return sum(set1 & set2) / sum(set1 | set2)
def sets_jaccard(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))


jaccard_generator1 = (1 - boolean_jaccard(row1, row2) for row1, row2 in combinations(df_interesse['covered'], r=2))
jaccard_generator2 = (1 - sets_jaccard(set(row1.selectors),
                                  set(row2.selectors)) for row1, row2 in combinations(df_interesse['subgroup'], r=2))
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


# Make the feature names shorter for the visualization
df_interesse.loc[:, 'subgroup'] = df_interesse['subgroup'].apply(lambda x: x.__str__().replace('sepal width (cm)', 'sw'))
df_interesse.loc[:, 'subgroup'] = df_interesse['subgroup'].apply(lambda x: x.__str__().replace('sepal length (cm)', 'sl'))
df_interesse.loc[:, 'subgroup'] = df_interesse['subgroup'].apply(lambda x: x.__str__().replace('petal width (cm)', 'pw'))
df_interesse.loc[:, 'subgroup'] = df_interesse['subgroup'].apply(lambda x: x.__str__().replace('petal length (cm)', 'pl'))

# plot the top three levels of the dendrogram
plt.figure()
plot_dendrogram(ac, truncate_mode="level", p=13, orientation='left',
                labels=list(df_interesse['subgroup']), leaf_font_size=8.)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlim(1, 0.45)
plt.tight_layout()
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
