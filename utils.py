import numpy as np
import pandas as pd
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


def extract_common_subgroups(df_dict):
    """
    Extract subgroups that appear in all three models' dataframes.

    Args:
        df_dict (dict): Dictionary with model names as keys and subgroup DataFrames as values

    Returns:
        pd.DataFrame: DataFrame containing only the subgroups that appear in all three models
    """
    # Create sets of subgroup descriptions for each model
    subgroup_sets = {}
    for model_name, df in df_dict.items():
        # Convert subgroups to strings for comparison
        subgroup_sets[model_name] = set(df['subgroup'].astype(str))

    # Find intersection of all sets (subgroups that appear in all models)
    common_subgroups = set.intersection(*subgroup_sets.values())

    # Create a list to store rows from each model for common subgroups
    common_rows = []

    # Extract rows for common subgroups from each model's dataframe
    for model_name, df in df_dict.items():
        # Filter rows where subgroup (as string) is in the common set
        model_common_rows = df[df['subgroup'].astype(str).isin(common_subgroups)].copy()

        # Add model name column
        model_common_rows['model'] = model_name

        # Add to the list of common rows
        common_rows.append(model_common_rows)

    # Combine all common rows into a single DataFrame
    if common_rows:
        result_df = pd.concat(common_rows, axis=0)
        # Sort by model name for better organization
        result_df = result_df.sort_values(by=['model'])
        return result_df
    else:
        # Return empty DataFrame with appropriate columns if no common subgroups
        return pd.DataFrame(columns=['model', 'quality', 'subgroup', 'size_sg', 'mean_sg', 'mean_dataset', 'covered'])
