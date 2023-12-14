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
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from venn import venn
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from functools import partial


def boxplot(errors_df: pd.DataFrame, target_names: list[str]) -> None:
    # Plot a boxplot with seaborn, using target names as labels
    errors_df.columns = target_names
    errors_df = errors_df.melt(var_name="class", value_name="error")
    sns.boxplot(x="class", y="error", data=errors_df)
    plt.show()
    # Print the average error for each class
    print(errors_df.groupby("class")["error"].mean())


# Define a custom quality function for a bidirectional search over the model's errors
class BidirectionalQFNumeric(ps.BoundedInterestingnessMeasure):
    tpl = namedtuple(
        "BidirectionalQFNumeric_parameters", ("size_sg", "mean", "estimate")
    )
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

    def calculate_statistics(
        self, subgroup, target, data, statistics=None
    ):  # pylint: disable=unused-argument
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

        def calculate_constant_statistics(
            self, data, target
        ):  # pylint: disable=unused-argument
            self.indices_greater_centroid = (
                self.qf.all_target_values
                > self.qf.read_centroid(self.qf.dataset_statistics)
            )
            self.target_values_greater_centroid = (
                self.qf.all_target_values
            )  # [self.indices_greater_mean]

        def get_estimate(
            self, subgroup, sg_size, sg_centroid, cover_arr, _
        ):  # pylint: disable=unused-argument
            larger_than_centroid = self.target_values_greater_centroid[cover_arr][
                self.indices_greater_centroid[cover_arr]
            ]
            size_greater_centroid = len(larger_than_centroid)
            sum_greater_centroid = np.sum(larger_than_centroid)

            return sum_greater_centroid - size_greater_centroid * self.qf.read_centroid(
                self.qf.dataset_statistics
            )


def subgroup_discovery(
    dataset_df: pd.DataFrame, errors_df: pd.DataFrame, number_of_classes: int
) -> dict:
    target = dataset_df.target.copy()
    dataset_df.drop("target", axis=1, inplace=True)

    X_sd = pd.concat([dataset_df, errors_df], axis=1)
    df_dict = {}
    for class_of_interest in range(number_of_classes):
        num_target = ps.NumericTarget(class_of_interest)
        searchspace = ps.create_selectors(X_sd, ignore=range(number_of_classes))
        task = ps.SubgroupDiscoveryTask(
            X_sd,
            num_target,
            searchspace,
            result_set_size=20,
            depth=2,
            qf=BidirectionalQFNumeric(a=0.5),
        )
        print("Mining relevant subgroups...")
        result = ps.BeamSearch().execute(task=task)
        df_regras = result.to_dataframe()
        df_regras["covered"] = df_regras["subgroup"].apply(lambda x: x.covers(X_sd))
        df_regras["class"] = class_of_interest
        df_dict[str(class_of_interest)] = df_regras.copy()
    dataset_df["target"] = target
    return df_dict


def latex_table(df_dict: dict, model: str) -> None:
    # Writing a LaTeX table with the subgroups found for one of the models
    df_print = df_dict[model].copy()
    df_print.loc[:, "subgroup"] = df_print["subgroup"].apply(
        lambda x: x.__str__().replace("sepal width (cm)", "sw")
    )
    df_print.loc[:, "subgroup"] = df_print["subgroup"].apply(
        lambda x: x.__str__().replace("sepal length (cm)", "sl")
    )
    df_print.loc[:, "subgroup"] = df_print["subgroup"].apply(
        lambda x: x.__str__().replace("petal width (cm)", "pw")
    )
    df_print.loc[:, "subgroup"] = df_print["subgroup"].apply(
        lambda x: x.__str__().replace("petal length (cm)", "pl")
    )
    print(
        df_print.to_latex(
            columns=["quality", "subgroup", "size_sg", "mean_sg"],
            header=["Quality", "Subgroup", "Size", "Average Error"],
            index=False,
            float_format="{:.2f}".format,
        )
    )


# Plot a Venn Diagram to compare which subgroups were identified as hard or easy for each of the models
def venn_diagram(df_regras: pd.DataFrame) -> None:
    set_list = []
    set_dict = {}
    for classe in df_regras["class"].unique():
        set_list.append(set(df_regras.loc[df_regras["class"] == classe, "subgroup"]))
        set_dict[classe] = set(df_regras.loc[df_regras["class"] == classe, "subgroup"])

    fig = go.Figure(go.Venn3(x=set_dict[0], y=set_dict[1]))


# Reducing the redundancy in the subgroups mined, by unifying the names of equal coverage subgroup descriptions
def remove_redundant_subgroups(df_dict):
    df_regras = pd.concat(df_dict, ignore_index=True)
    lista_regras = []
    lista_intersecoes = []

    # First, we create a list of the subgroup descriptions and its coverage sets
    for _, linha_regra in tqdm(df_regras.iterrows()):
        perc_igual = 0
        for r in lista_regras:
            perc_igual = sum(linha_regra["covered"] & r["covered"]) / sum(
                linha_regra["covered"] | r["covered"]
            )
            lista_intersecoes.append(perc_igual)
            if perc_igual == 1:
                r["subgroup"].add(linha_regra["subgroup"])
                break
        if perc_igual != 1:
            lista_regras.append(
                {
                    "subgroup": {linha_regra["subgroup"]},
                    "covered": linha_regra["covered"],
                }
            )

    # Then, we create a dictionary to replace all descriptions by the first of them
    regras_repetidas = [x["subgroup"] for x in lista_regras if len(x["subgroup"]) > 1]
    mapa_regras = {}
    for conjunto in regras_repetidas:
        primeira_regra = None
        for r in conjunto:
            if primeira_regra is None:
                primeira_regra = r
            else:
                mapa_regras[r] = primeira_regra

    """ df_regras = df_regras.replace(
        {
            "subgroup": mapa_regras,
            "class": {0: "setosa", 1: "versicolor", 2: "virginica"},
        }
    ) """

    df_regras = df_regras.replace(
        {
            "subgroup": mapa_regras,
        }
    )
    return df_regras


# Plot a 2D scatterplot showing the samples, its classes, and the subgroups passed as parameters
# to the functio
def plot_subgroups_px(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    target: Iterable,
    subgroups: pd.DataFrame,
):  
    
    classes = map(lambda x : "setosa" if x == 0 else "versicolor" if x == 1 else "virginica", target)
    fig = px.scatter(data, x=x_column, y=y_column, color=[str(x) for x in classes])

    # if subgroup is None, plot only the data
    if subgroups is None:
        return fig

    rectangle_line_width = 2.0

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
                raise (
                    NotImplementedError("I still can't deal with non numeric features!")
                )
            if isinstance(rule2, ps.IntervalSelector):
                rule2_upperbound = rule2.upper_bound
                rule2_lowerbound = rule2.lower_bound
                rule2_attribute = rule2.attribute_name
                if rule2_lowerbound == float("-inf"):
                    rule2_lowerbound = data[rule2_attribute].min()
                if rule2_upperbound == float("inf"):
                    rule2_upperbound = data[rule2_attribute].max()
            else:
                raise (
                    NotImplementedError("I still can't deal with non numeric features!")
                )
        # draw a red or green rectangle around the region of interest
        if subgroup.mean_sg > subgroup.mean_dataset:
            color = "red"
        else:
            color = "green"
        if rule1_attribute == x_column:
            x0 = rule1_lowerbound - delta
            y0 = rule2_lowerbound - delta
            width = rule1_upperbound - rule1_lowerbound + 2 * delta
            height = rule2_upperbound - rule2_lowerbound + 2 * delta
            fig.add_shape(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x0 + width,
                y1=y0 + height,
                line=dict(color=color, width=rectangle_line_width),
            )
            fig.add_annotation(
                x=x0,
                y=y0,
                text=round(subgroup.mean_sg, 4),
                showarrow=False,
                xanchor="right",
                yanchor="top",
            )

            fig.add_annotation(
                x=x0 + width,
                y=y0 + height,
                text=round(subgroup.mean_dataset, 4),
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
            )
        else:
            x0 = rule2_lowerbound - delta
            y0 = rule1_lowerbound - delta
            width = rule2_upperbound - rule2_lowerbound + 2 * delta
            height = rule1_upperbound - rule1_lowerbound + 2 * delta
            fig.add_shape(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x0 + width,
                y1=y0 + height,
                line=dict(color=color, width=rectangle_line_width),
            )
            fig.add_annotation(
                x=x0,
                y=y0,
                text=round(subgroup.mean_sg, 4),
                showarrow=False,
                xanchor="right",
                yanchor="top",
            )
            fig.add_annotation(
                x=x0 + width,
                y=y0 + height,
                text=round(subgroup.mean_dataset, 4),
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
            )
    return fig


# Pre-calculating the Jaccard Index
def boolean_jaccard(set1, set2):
    return sum(set1 & set2) / sum(set1 | set2)


def sets_jaccard(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))


def plot_dendrogram(df_regras: pd.DataFrame):
    set_dict = {}

    df_regras = df_regras.replace(
        {"class": {0: "setosa", 1: "versicolor", 2: "virginica"}}
    )

    for classe in df_regras["class"].unique():
        set_dict[classe] = set(df_regras.loc[df_regras["class"] == classe, "subgroup"])
    regras_interesse = (
        set_dict["versicolor"].union(set_dict["virginica"]) - set_dict["setosa"]
    )
    df_interesse = df_regras.loc[
        df_regras["subgroup"].isin(regras_interesse), ["subgroup", "covered"]
    ]
    df_interesse.drop_duplicates(subset="subgroup", inplace=True)
    # Create linkage matrix and then plot the dendrogram

    jaccard_generator1 = (
        1 - boolean_jaccard(row1, row2)
        for row1, row2 in combinations(df_interesse["covered"], r=2)
    )
    jaccard_generator2 = (
        1 - sets_jaccard(set(row1.selectors), set(row2.selectors))
        for row1, row2 in combinations(df_interesse["subgroup"], r=2)
    )
    flattened_matrix1 = np.fromiter(jaccard_generator1, dtype=np.float64)
    flattened_matrix2 = np.fromiter(jaccard_generator2, dtype=np.float64)
    flattened_matrix = np.minimum(flattened_matrix1, flattened_matrix2)

    # since flattened_matrix is the flattened upper triangle of the matrix
    # we need to expand it.
    normal_matrix = distance.squareform(flattened_matrix)
    # replacing zeros with ones at the diagonal.
    # normal_matrix += np.identity(len(df_interesse['covered']))

    # setting distance_threshold=0 ensures we compute the full tree.
    ac = AgglomerativeClustering(
        distance_threshold=0, n_clusters=None, metric="precomputed", linkage="average"
    )
    ac.fit(normal_matrix)

    # create the counts of samples under each node
    counts = np.zeros(ac.children_.shape[0])
    n_samples = len(ac.labels_)
    for i, merge in enumerate(ac.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([ac.children_, ac.distances_, counts]).astype(
        float
    )

    # Make the feature names shorter for the visualization
    df_interesse.loc[:, "subgroup"] = df_interesse["subgroup"].apply(
        lambda x: x.__str__().replace("sepal width (cm)", "sw")
    )
    df_interesse.loc[:, "subgroup"] = df_interesse["subgroup"].apply(
        lambda x: x.__str__().replace("sepal length (cm)", "sl")
    )
    df_interesse.loc[:, "subgroup"] = df_interesse["subgroup"].apply(
        lambda x: x.__str__().replace("petal width (cm)", "pw")
    )
    df_interesse.loc[:, "subgroup"] = df_interesse["subgroup"].apply(
        lambda x: x.__str__().replace("petal length (cm)", "pl")
    )

    # Plot the corresponding dendrogram

    # VER COM DANIEL QUESTÃO DA LINKAGE MATRIX
    fig = ff.create_dendrogram(
        X=normal_matrix,
        orientation="right",
        labels=df_interesse.subgroup.tolist(),
        linkagefun=lambda _: linkage_matrix,
        
    )

    # 1800 width fitted well on my screen, but it should be more dynamic
    fig.update_layout(width=1000, height=600, yaxis={"side": "right"})
    fig.update_xaxes(range=[-1, -0.45])

    return fig
