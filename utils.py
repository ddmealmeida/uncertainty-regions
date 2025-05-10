import numpy as np
from scipy.cluster.hierarchy import dendrogram

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


def calculate_coverage_metrics(data_frame, high_error, coverage_masks=None):
    """
    Calculate cumulative coverage and precision metrics for subgroup detection.

    Parameters:
    -----------
    data_frame : Union[pandas.DataFrame, list]
        DataFrame containing the subgroups and their coverage information,
        or list of slices for slice finder
    high_error : numpy.ndarray
        Boolean array indicating which samples have high error
    coverage_masks : numpy.ndarray, optional
        Array of coverage masks for slice finder method (default=None)

    Returns:
    --------
    tuple
        (cumulative_coverage, cumulative_precision) where each is a list containing
        the metrics calculated for each number of subgroups
    """
    # Get length from the first coverage mask
    if coverage_masks is not None:
        length = len(coverage_masks[0])
    else:
        length = len(data_frame.iloc[0]['covered'])

    coverage_array = np.full((length,), False)
    cumulative_coverage = []
    cumulative_precision = []

    n_iterations = len(data_frame)

    for n_group in range(n_iterations):
        if coverage_masks is not None:
            coverage_array = coverage_array | coverage_masks[n_group]
        else:
            coverage_array = coverage_array | data_frame.iloc[n_group]['covered']

        new_coverage = sum(coverage_array[high_error]) / sum(high_error)
        cumulative_coverage.append(new_coverage)
        new_precision = sum(coverage_array[high_error]) / sum(coverage_array)
        cumulative_precision.append(new_precision)

    return cumulative_coverage, cumulative_precision


def convert_binary_columns(df):
    """Convert all binary categorical columns in a DataFrame to 0/1 values.

    Args:
        df (pandas.DataFrame): Input DataFrame containing binary categorical columns.

    Returns:
        pandas.DataFrame: DataFrame with binary columns converted to 0/1 values.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Identify binary categorical columns (those with only 2 unique values)
    binary_columns = df_copy.select_dtypes(include=['object']).columns[
        df_copy.select_dtypes(include=['object']).nunique() == 2
        ]

    # Create a mapping dictionary for each binary column
    for col in binary_columns:
        unique_values = df_copy[col].unique()
        # Create a mapping where first unique value is 0, second is 1
        mapping = {unique_values[0]: 0, unique_values[1]: 1}
        # Apply the mapping to convert to 0/1
        df_copy[col] = df_copy[col].map(mapping)

    return df_copy
