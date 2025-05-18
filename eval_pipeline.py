import numpy as np
import pandas as pd
import seaborn as sns
import pysubgroup as ps
import matplotlib.pyplot as plt
from typing import Iterable
from itertools import combinations
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from venn import venn

from utils import boolean_jaccard, sets_jaccard, plot_dendrogram


def run_subgroup_discovery(features: pd.DataFrame,
                           errors: pd.Series,
                           a: float = 0.5,
                           result_size: int = 20,
                           nbins: int = 5):
    """
    Perform subgroup discovery to identify patterns in the dataset that are associated 
    with high prediction errors.

    Args:
        features (pd.DataFrame): A DataFrame containing the feature set to evaluate subgroups.
                                 Each row represents an instance, and columns represent feature dimensions.
        errors (pd.Series): A Series representing the prediction errors corresponding to each row of the
                            `features` DataFrame. The values are numeric.
        a (float): The exponent of the quality function, a trade-off between unusualness and size of the subgroup
        result_size (int): The number of returned subgroups. Default is 20.
        nbins (int): The number of bins to use for the discretization of continuous features. Default is 5.

    Returns:
        pd.DataFrame: A DataFrame containing the discovered subgroups. Each row corresponds to a subgroup, 
                      with the following columns:
                      - 'subgroup': The subgroup description (e.g., feature-based rules).
                      - 'quality': The quality metric associated with the subgroup.
                      - 'size_sg': The size of the subgroup (number of instances it contains).
                      - 'mean_sg': The average error value within the subgroup.
                      - 'covered': A boolean Series indicating which instances are covered by the subgroup.
    """
    df = features.copy()
    df.loc[:, 'error'] = errors.copy()

    num_target = ps.NumericTarget('error')
    searchspace = ps.create_selectors(df, ignore=['error'], nbins=nbins)
    task = ps.SubgroupDiscoveryTask(
        df,
        num_target,
        searchspace,
        result_set_size=result_size,
        depth=2,
        qf=ps.StandardQFNumeric(a=a))
    print('Mining relevant subgroups...')
    result = ps.BeamSearch().execute(task=task)
    df_subgroup = result.to_dataframe()
    df_subgroup['covered'] = df_subgroup['subgroup'].apply(lambda x: x.covers(df))
    return df_subgroup


def run_hierarchical_clustering(df: pd.DataFrame):
    """
    Perform hierarchical clustering on the discovered subgroups to analyze the similarity 
    between them using a combination of boolean and set-based Jaccard distances, and plot a dendrogram of the results.

    Args:
        df (pd.DataFrame): A DataFrame containing information about the discovered subgroups, 
                           including a 'covered' column that represents the covered instances 
                           and a 'subgroup' column with subgroup descriptions.

    Returns:
        AgglomerativeClustering: The fitted Agglomerative Clustering model after performing
                                 hierarchical clustering on the subgroups.
    """
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

    # setting distance_threshold=0 ensures we compute the full tree.
    ac = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='average')
    ac.fit(normal_matrix)

    # Shorten the feature names for the visualization
    df['subgroup_short'] = df['subgroup'].apply(lambda x: x.__str__().replace('sepal width (cm)', 'sw'))
    df.loc[:, 'subgroup_short'] = df['subgroup_short'].apply(lambda x: x.__str__().replace('sepal length (cm)', 'sl'))
    df.loc[:, 'subgroup_short'] = df['subgroup_short'].apply(lambda x: x.__str__().replace('petal width (cm)', 'pw'))
    df.loc[:, 'subgroup_short'] = df['subgroup_short'].apply(lambda x: x.__str__().replace('petal length (cm)', 'pl'))

    # plot the dendrogram
    plt.figure()
    plot_dendrogram(ac, truncate_mode="level", p=13, orientation='left',
                    labels=list(df['subgroup_short']), leaf_font_size=8.)
    plt.title("                      Hierarchical Clustering Dendrogram")
    plt.xlim(1, min(ac.distances_))
    plt.tight_layout()
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    return ac

def filter_redundant_subgroups(df: pd.DataFrame, ac: AgglomerativeClustering,
                               distance_threshold: float) -> pd.DataFrame:
    """
    Filter redundant subgroups from the result of hierarchical clustering based on a distance threshold.

    Args:
        df (pd.DataFrame): A DataFrame containing information about discovered subgroups,
                           including columns such as 'quality' and 'subgroup'.
        ac (AgglomerativeClustering): A fitted AgglomerativeClustering model used for hierarchical clustering.
        distance_threshold (float): The maximum linkage distance below which clusters are considered redundant.

    Returns:
        pd.DataFrame: A DataFrame containing only the relevant subgroups.
    """
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
    return df.loc[subgroup_filter].copy()

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

    # Add jitter to columns that have up to 10 unique values
    jittered_data = data.copy()
    for col in data.columns:
        if data[col].nunique() <= 20:
            jittered_data[col] = data[col] + (np.random.random(len(data)) - 0.5) * data[col].std()/10

    sns.scatterplot(data=jittered_data,
                    x=x_column,
                    y=y_column,
                    hue=target,
                    s=20, alpha=0.5, ax=ax)
    # Rectangle displacement, estimated from my head
    mean_delta = 0.02
    for subgroup in subgroups.itertuples(index=False):
        delta = mean_delta + np.random.random()/50
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
            # ax.text(rule1_upperbound, rule2_upperbound, round(subgroup.mean_dataset, 4), fontsize=8)
        else:
            ax.add_patch(plt.Rectangle((rule2_lowerbound - delta, rule1_lowerbound - delta),
                                       width=rule2_upperbound - rule2_lowerbound + 2*delta,
                                       height=rule1_upperbound - rule1_lowerbound + 2*delta,
                                       fill=False, edgecolor=color, linewidth=1))
            ax.text(rule2_lowerbound, rule1_lowerbound, round(subgroup.mean_sg, 4), fontsize=8)
            # ax.text(rule2_upperbound, rule1_upperbound, round(subgroup.mean_dataset, 4), fontsize=8)
    return ax

def run_subgroup_comparison(df_dict: dict):
    set_list = []
    set_dict = {}
    for model, df in df_dict.items():
        set_list.append(set(df['subgroup']))
        set_dict[model] = set(df['subgroup'])
    venn(set_dict)
    plt.show()


def plot_subgroups_with_zoom(data: pd.DataFrame,
                             x_column: str,
                             y_column: str,
                             target: Iterable,
                             subgroups: pd.DataFrame,
                             figsize=(12, 5),
                             subplot_ratio=0.5):
    """
    Plot a 2D scatterplot showing the samples, its classes, and the subgroups passed as parameters,
    along with a zoomed-in view of the data points inside each subgroup.

    Args:
        data (pd.DataFrame): DataFrame containing the data to plot
        x_column (str): Column name for x-axis
        y_column (str): Column name for y-axis
        target (Iterable): Values for the hue parameter (classes)
        subgroups (pd.DataFrame): DataFrame containing subgroup information
        figsize (tuple): Figure size (width, height)
        subplot_ratio (float): Ratio between main plot and zoom plot widths

    Returns:
        fig, (ax1, ax2): Figure and axes objects
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={'width_ratios': [1 - subplot_ratio, subplot_ratio]})

    # Plot the main view with all data points and subgroups
    ax1 = plot_subgroups(data, x_column, y_column, target, subgroups, ax=ax1)
    ax1.set_title("Full View")

    # Add jitter to columns that have up to 10 unique values
    jittered_data = data.copy()
    for col in data.columns:
        if data[col].nunique() <= 20:
            jittered_data[col] = data[col] + (np.random.random(len(data)) - 0.5) * data[col].std() / 10

    # Find the boundaries of the subgroups to create a zoomed-in view
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')

    # Keep track of points inside any subgroup for coloring
    points_in_subgroups = np.zeros(len(data), dtype=bool)

    # Extract boundaries for each subgroup
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
                # Update x boundaries
                x_min = min(x_min, rule1_lowerbound)
                x_max = max(x_max, rule1_upperbound)
                # Use full range for y
                rule2_lowerbound = data[rule2_attribute].min()
                rule2_upperbound = data[rule2_attribute].max()
                # Update y boundaries
                y_min = min(y_min, rule2_lowerbound)
                y_max = max(y_max, rule2_upperbound)
                # Mark points in this subgroup
                points_in_subgroup = (data[rule1_attribute] >= rule1_lowerbound) & \
                                     (data[rule1_attribute] <= rule1_upperbound)
            else:
                rule2_attribute = x_column
                # Use full range for x
                rule2_lowerbound = data[rule2_attribute].min()
                rule2_upperbound = data[rule2_attribute].max()
                # Update x boundaries
                x_min = min(x_min, rule2_lowerbound)
                x_max = max(x_max, rule2_upperbound)
                # Update y boundaries
                y_min = min(y_min, rule1_lowerbound)
                y_max = max(y_max, rule1_upperbound)
                # Mark points in this subgroup
                points_in_subgroup = (data[rule1_attribute] >= rule1_lowerbound) & \
                                     (data[rule1_attribute] <= rule1_upperbound)
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
                raise (NotImplementedError("I still can't deal with non numeric features!"))

            if isinstance(rule2, ps.IntervalSelector):
                rule2_upperbound = rule2.upper_bound
                rule2_lowerbound = rule2.lower_bound
                rule2_attribute = rule2.attribute_name
                if rule2_lowerbound == float("-inf"):
                    rule2_lowerbound = data[rule2_attribute].min()
                if rule2_upperbound == float("inf"):
                    rule2_upperbound = data[rule2_attribute].max()
            else:
                raise (NotImplementedError("I still can't deal with non numeric features!"))

            if rule1_attribute == x_column:
                # Update x boundaries
                x_min = min(x_min, rule1_lowerbound)
                x_max = max(x_max, rule1_upperbound)
                # Update y boundaries
                y_min = min(y_min, rule2_lowerbound)
                y_max = max(y_max, rule2_upperbound)
                # Mark points in this subgroup
                points_in_subgroup = (data[rule1_attribute] >= rule1_lowerbound) & \
                                     (data[rule1_attribute] <= rule1_upperbound) & \
                                     (data[rule2_attribute] >= rule2_lowerbound) & \
                                     (data[rule2_attribute] <= rule2_upperbound)
            else:
                # Update x boundaries
                x_min = min(x_min, rule2_lowerbound)
                x_max = max(x_max, rule2_upperbound)
                # Update y boundaries
                y_min = min(y_min, rule1_lowerbound)
                y_max = max(y_max, rule1_upperbound)
                # Mark points in this subgroup
                points_in_subgroup = (data[rule1_attribute] >= rule1_lowerbound) & \
                                     (data[rule1_attribute] <= rule1_upperbound) & \
                                     (data[rule2_attribute] >= rule2_lowerbound) & \
                                     (data[rule2_attribute] <= rule2_upperbound)

        points_in_subgroups = points_in_subgroups | points_in_subgroup

    # Add some padding to the zoom view (10% on each side)
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    # Plot only the data points inside any subgroup in the zoom view
    zoomed_data = jittered_data[points_in_subgroups].copy()
    if len(zoomed_data) > 0:
        # Make sure to use the same indices for target to match the filtered data
        if isinstance(target, pd.Series):
            zoomed_target = target.loc[zoomed_data.index]
        elif isinstance(target, np.ndarray) and len(target) == len(data):
            # If target is a numpy array, we need to filter it using the same boolean mask
            zoomed_target = target[points_in_subgroups]
        else:
            # For other cases, try to use the same indices
            zoomed_target = target

        # Plot zoomed data
        sns.scatterplot(data=zoomed_data,
                        x=x_column,
                        y=y_column,
                        hue=zoomed_target,
                        s=50, alpha=0.7, ax=ax2)

        # Set the boundaries for the zoom view
        ax2.set_xlim(x_min - x_padding, x_max + x_padding)
        ax2.set_ylim(y_min - y_padding, y_max + y_padding)
        ax2.set_title("Zoomed View of Subgroup")

        # Draw the subgroup boundaries in the zoomed view as well
        mean_delta = 0.02
        for subgroup in subgroups.itertuples(index=False):
            delta = mean_delta + np.random.random() / 50
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
                    raise (NotImplementedError("I still can't deal with non numeric features!"))
                if isinstance(rule2, ps.IntervalSelector):
                    rule2_upperbound = rule2.upper_bound
                    rule2_lowerbound = rule2.lower_bound
                    rule2_attribute = rule2.attribute_name
                    if rule2_lowerbound == float("-inf"):
                        rule2_lowerbound = data[rule2_attribute].min()
                    if rule2_upperbound == float("inf"):
                        rule2_upperbound = data[rule2_attribute].max()
                else:
                    raise (NotImplementedError("I still can't deal with non numeric features!"))

            # Draw rectangles in the zoomed view with the same color logic
            if subgroup.mean_sg > subgroup.mean_dataset:
                color = 'red'
            else:
                color = 'green'

            if rule1_attribute == x_column:
                ax2.add_patch(plt.Rectangle((rule1_lowerbound - delta, rule2_lowerbound - delta),
                                            width=rule1_upperbound - rule1_lowerbound + 2 * delta,
                                            height=rule2_upperbound - rule2_lowerbound + 2 * delta,
                                            fill=False, edgecolor=color, linewidth=1))
                ax2.text(rule1_lowerbound, rule2_lowerbound, round(subgroup.mean_sg, 4), fontsize=8)
            else:
                ax2.add_patch(plt.Rectangle((rule2_lowerbound - delta, rule1_lowerbound - delta),
                                            width=rule2_upperbound - rule2_lowerbound + 2 * delta,
                                            height=rule1_upperbound - rule1_lowerbound + 2 * delta,
                                            fill=False, edgecolor=color, linewidth=1))
                ax2.text(rule2_lowerbound, rule1_lowerbound, round(subgroup.mean_sg, 4), fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No data points in subgroups",
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes)
        ax2.set_title("No Data in Subgroup")

    plt.tight_layout()
    return fig, (ax1, ax2)
